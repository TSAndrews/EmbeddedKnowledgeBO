import os
import time
import torch
from torch import Tensor
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.transforms import unnormalize, normalize
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch import fit_gpytorch_mll
from botorch.acquisition.multi_objective import MCMultiOutputObjective
from typing import Callable, Optional
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.constraints import Interval

SMOKE_TEST = os.environ.get("SMOKE_TEST")

NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4
    

def generate_initial_sample(problem,n=6):
    # generate training data
    t0 = time.monotonic()
    train_x = draw_sobol_samples(bounds=problem.bounds, n=n, q=1).squeeze(1)
    t1 = time.monotonic()
    return train_x, t1-t0

def get_observation(train_x,problem,noise=0):
    train_obj_true = problem(train_x)
    train_obj = train_obj_true + torch.randn_like(train_obj_true) * noise
    return train_obj

def initialize_model(train_x, train_obj,problem,noise=None,covar_module=None):#,subset_batch_dict=None):
    # define models for objective and constraint
    train_x = normalize(train_x, problem.bounds)
    models = []
    for i in range(train_obj.shape[-1]):
        train_y = train_obj[..., i : i + 1]
        train_yvar = torch.full_like(train_y, noise[i] ** 2) if noise is not None else None
        m=SingleTaskGP(train_x, train_y, train_yvar, outcome_transform=Standardize(m=1))#,covar_module=covar_module)
        if covar_module is not None:
            m.covar_module=covar_module
            # if subset_batch_dict is None: raise Exception("subset_batch_dict must be defined if a custom covariance kernel is provided as covar_module")
            # m._subset_batch_dict=subset_batch_dict
        models.append(m)
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model

def optimize_qehvi(model, train_x, problem, sampler, batch_size=1,tkwargs={"dtype": torch.double,"device": torch.device("cuda" if torch.cuda.is_available() else "cpu")},**kwargs):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    standard_bounds = torch.zeros(2, problem.dim, **tkwargs)
    standard_bounds[1] = 1    
    post_transform=kwargs.get("objective")
    # partition non-dominated space into disjoint rectangles
    with torch.no_grad():
        if post_transform is not None:
            pred = post_transform.objective(model.posterior(normalize(train_x, problem.bounds)).mean,normalize(train_x, problem.bounds))
        else:
            pred = model.posterior(normalize(train_x, problem.bounds)).mean
    partitioning = FastNondominatedPartitioning(ref_point=problem.ref_point,Y=pred,)
    acq_func = qExpectedHypervolumeImprovement(model=model,ref_point=problem.ref_point,partitioning=partitioning,sampler=sampler,**kwargs)
    # optimize
    candidates, _ = optimize_acqf(acq_function=acq_func,bounds=standard_bounds, q=batch_size,num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES,options={"batch_limit": 5, "maxiter": 200},sequential=True,)
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    return new_x

def get_recomendations(train_x, train_obj, problem, model_initializer,acquisition_func=optimize_qehvi,batch_size=1,mc_samples=1,model_initializer_kwargs={},acq_kwargs={},tkwargs = {"dtype": torch.double,"device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}):
    t0 = time.monotonic()
    dims=list(train_x.shape)
    dims[0]=0
    x_qehvi=torch.empty(dims)
    
    # initialize the models so they are ready for fitting on next iteration
    mll_qehvi, model_qehvi = model_initializer(train_x, train_obj.squeeze(0),problem,**model_initializer_kwargs)

    # fit the models
    fit_gpytorch_mll(mll_qehvi)

    # define the qEI and qNEI acquisition modules using a QMC sampler
    qehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))

    # optimize acquisition functions and get new observations
    new_x_qehvi = acquisition_func(model_qehvi, train_x, problem, qehvi_sampler,batch_size=batch_size,tkwargs=tkwargs,**acq_kwargs)
        
    x_qehvi=torch.cat([x_qehvi, new_x_qehvi])

    t1 = time.monotonic()
            
    return x_qehvi, t1-t0

class GenericMCMultiObjective(MCMultiOutputObjective):
    r"""Literaly a copy of GenericMCObjective but inheriting from MCMultiOutputObjective
    Objective generated from a generic callable.

    Allows to construct arbitrary MC-objective functions from a generic
    callable. In order to be able to use gradient-based acquisition function
    optimization it should be possible to backpropagate through the callable.

    Example:
        >>> generic_objective = GenericMCObjective(
                lambda Y, X: torch.sqrt(Y).sum(dim=-1),
            )
        >>> samples = sampler(posterior)
        >>> objective = generic_objective(samples)
    """

    def __init__(self, objective: Callable[[Tensor, Optional[Tensor]], Tensor]) -> None:
        r"""
        Args:
            objective: A callable `f(samples, X)` mapping a
                `sample_shape x batch-shape x q x m`-dim Tensor `samples` and
                an optional `batch-shape x q x d`-dim Tensor `X` to a
                `sample_shape x batch-shape x q`-dim Tensor of objective values.
        """
        super().__init__()
        self.objective = objective

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        r"""Evaluate the objective on the samples.

        Args:
            samples: A `sample_shape x batch_shape x q x m`-dim Tensors of
                samples from a model posterior.
            X: A `batch_shape x q x d`-dim tensor of inputs. Relevant only if
                the objective depends on the inputs explicitly.

        Returns:
            A `sample_shape x batch_shape x q`-dim Tensor of objective values.
        """
        return self.objective(samples, X=X)
    
def objective_factory(objectives,problem):
    """"""
    def objective(Y,X):
        X_raw=unnormalize(X,problem.bounds)
        Ynew=[obj(Y,X_raw) for obj in objectives]
        return torch.cat(Ynew,dim=-1)
    return GenericMCMultiObjective(objective)

def get_matern_kernel_with_bounded_gamma_prior(ard_num_dims: int, batch_shape: Optional[torch.Size] = None, lengthscale_constraint: Optional[Interval] = None) -> ScaleKernel:
    r"""Constructs the Scale-Matern kernel. Uses a Gamma(3.0, 6.0) prior for the lengthscale bounded by lengthscale_constraint
    and a Gamma(2.0, 0.15) prior for the output scale.
    """
    return ScaleKernel(
        base_kernel=MaternKernel(
            nu=2.5,
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
            lengthscale_prior=GammaPrior(3.0, 6.0),
            lengthscale_constraint=lengthscale_constraint
        ),
        batch_shape=batch_shape,
        outputscale_prior=GammaPrior(2.0, 0.15),
    )
