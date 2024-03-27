import os
import torch
import time
from botorch.utils.transforms import unnormalize, normalize
from src.optimization_utils import generate_initial_sample, get_observation, optimize_qehvi, get_recomendations
from botorch.acquisition.objective import GenericMCObjective
import warnings
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize
from typing import Optional
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.constraints import Interval
#from botorch.utils.multi_objective.pareto import is_non_dominated


warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")
NOISE_SE = torch.tensor([15.19, 0.63], **tkwargs)

# ### Problem setup
# from botorch.test_functions.multi_objective import BraninCurrin
# problem = BraninCurrin(negate=True).to(**tkwargs)
from src.benchmark import SnArModel
problem = SnArModel().to(**tkwargs)

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

def initialize_model_fitted_noise(train_x, train_obj,problem,noise,covar_module):
    # define models for objective and constraint
    train_x = normalize(train_x, problem.bounds)
    models = []
    for i in range(train_obj.shape[-1]):
        train_y = train_obj[..., i : i + 1]
        train_yvar = torch.full_like(train_y, noise[i] ** 2)
        models.append(SingleTaskGP(train_x, train_y, train_yvar, covar_module=covar_module,outcome_transform=Standardize(m=1)))
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model

def compute_space_time_yield(Y,X):
    res_time=X[:,:,0].unsqueeze(-1)
    sty=Y/res_time
    return sty[:,:,:,-1]

def compute_e_factor(Y,X):
    equiv_pldn=X[:,:,1].unsqueeze(-1)
    conc_dfnb=X[:,:,2].unsqueeze(-1)
    
    rho_eth = 789  # g/L (should adjust to temp, but just using @ 25C)
    
    mass_reagents=(rho_eth + conc_dfnb*159.09 + equiv_pldn*conc_dfnb*71.11)#g/L. *1e3*q_tot = g/min 
    mass_products=(Y*210.21)#g/L. *1e3*q_tot = g/min
    e_factor = mass_reagents / mass_products
    return -e_factor[:,:,:,-1] #minus to convert to maximization

def objective_factory(objectives):
    """objectives (list[Callable]) - list of objective functions to compute.
        objective (Callable[[Tensor, Tensor], Tensor]) – A callable f(samples, X) mapping a sample_shape x batch-shape x q x m-dim Tensor samples
                and an optional batch-shape x q x d-dim Tensor X to a sample_shape x batch-shape x q-dim Tensor of objective values."""
    def objective(Y,X):
        X_raw=unnormalize(X,problem.bounds)
        Ynew=[obj(Y,X_raw) for obj in objectives]
        return torch.cat(Ynew,dim=-1)
    return GenericMCObjective(objective)

acquisition_objective=objective_factory([compute_space_time_yield,compute_e_factor])

iterations=20
batch_size=1
mc_samples = 128
train_size=2 * (problem.dim + 1)
num_model_outputs=torch.Size([1])
min_lengthscale=0.1

kernel=get_matern_kernel_with_bounded_gamma_prior(problem.dim,num_model_outputs,lengthscale_constraint=Interval(lower_bound=min_lengthscale))

# call helper functions to generate initial training data and initialize model
train_x_qehvi, t0 = generate_initial_sample(problem,n=train_size)
train_obj_qehvi = get_observation(train_x_qehvi,problem)

iter_time=[t0/train_size]*train_size
for _ in range(iterations):
    x_new,t=get_recomendations(train_x_qehvi,train_obj_qehvi,problem,model_initializer=initialize_model_fitted_noise,acquisition_func=optimize_qehvi, batch_size=batch_size,mc_samples=mc_samples,model_initializer_kwargs={"covar_module":kernel})
    obj_new=get_observation(x_new,problem)
    
    train_x_qehvi=torch.cat([train_x_qehvi,x_new])
    train_obj_qehvi=torch.cat([train_obj_qehvi,obj_new])
    iter_time.append(t)
    
hypervolume=[]
for i in range(train_x_qehvi.shape[0]):
    bd = DominatedPartitioning(ref_point=problem.ref_point, Y=problem(train_x_qehvi[0:i]))
    hypervolume.append(bd.compute_hypervolume().item())
 
import matplotlib.pyplot as plt
plt.plot(hypervolume)
plt.show()

beta^alpha / Gamma(alpha) * x^(alpha - 1) * exp(-beta * x)