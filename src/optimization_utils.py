import os
import time
import torch
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.transforms import unnormalize, normalize
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch import fit_gpytorch_mll

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

def optimize_qehvi(model, train_x, problem, sampler, batch_size=1,tkwargs={"dtype": torch.double,"device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    standard_bounds = torch.zeros(2, problem.dim, **tkwargs)
    standard_bounds[1] = 1    

    # partition non-dominated space into disjoint rectangles
    with torch.no_grad():
        pred = model.posterior(normalize(train_x, problem.bounds)).mean
    partitioning = FastNondominatedPartitioning(ref_point=problem.ref_point,Y=pred,)
    acq_func = qExpectedHypervolumeImprovement(model=model,ref_point=problem.ref_point,partitioning=partitioning,sampler=sampler,)
    # optimize
    candidates, _ = optimize_acqf(acq_function=acq_func,bounds=standard_bounds, q=batch_size,num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES,options={"batch_limit": 5, "maxiter": 200},sequential=True,)
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    return new_x

def get_recomendations(train_x, train_obj, problem, model_initializer,acquisition_func=optimize_qehvi,batch_size=1,mc_samples=1,model_initializer_kwargs={}):
    t0 = time.monotonic()
    dims=list(train_x.shape)
    dims[0]=0
    x_qehvi=torch.empty(dims)
    
    # initialize the models so they are ready for fitting on next iteration
    mll_qehvi, model_qehvi = model_initializer(train_x, train_obj,problem,**model_initializer_kwargs)

    # fit the models
    fit_gpytorch_mll(mll_qehvi)

    # define the qEI and qNEI acquisition modules using a QMC sampler
    qehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))

    # optimize acquisition functions and get new observations
    new_x_qehvi = acquisition_func(model_qehvi, train_x, problem, qehvi_sampler,batch_size=batch_size,tkwargs=tkwargs)
        
    x_qehvi=torch.cat([x_qehvi, new_x_qehvi])

    t1 = time.monotonic()
            
    return x_qehvi, t1-t0