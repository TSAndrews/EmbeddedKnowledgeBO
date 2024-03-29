import os
import torch
import time
from botorch.utils.transforms import unnormalize, normalize
from src.optimization_utils import generate_initial_sample,objective_factory, initialize_model, get_matern_kernel_with_bounded_gamma_prior
from gpytorch.constraints import Interval
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
import warnings
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
import matplotlib.pyplot as plt
import numpy as np
#from botorch.utils.multi_objective.pareto import is_non_dominated


warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")

# ### Problem setup
# from botorch.test_functions.multi_objective import BraninCurrin
# problem = BraninCurrin(negate=True).to(**tkwargs)
from src.benchmark import SnArModel, compute_space_time_yield,compute_e_factor
problem = SnArModel(noise_std=0.01).to(**tkwargs)

#Space time yield posteria against residence time: pre vs post objective computation
from botorch.test_functions.base import BaseTestProblem
class Dummy(BaseTestProblem):
    _bounds=[problem._bounds[0]]
    dim=1
    def evaluate_true(self, X):
        return X[...,0]
p=Dummy().to(**tkwargs)
n_train=100
n_test=100
f, (ax1,ax2) = plt.subplots(1, 2, figsize=(6, 4))
torch.manual_seed(1)
train_x_normalized=0.5*torch.ones((n_train,4))
train_x_normalized[:,0]=torch.rand((n_train))
train_x=unnormalize(train_x_normalized,problem.bounds)
train_yield=problem(train_x)

lengthscales=[]
r2=[]
noises=[]
for length in np.linspace(50,100,n_test):
    lengthscales.append(length)
    kernel=get_matern_kernel_with_bounded_gamma_prior(p.dim,lengthscale_constraint=Interval(lower_bound=length,upper_bound=length+1/(2*n_test)))
    mll_yield,model_yield=initialize_model(train_x[:,0].unsqueeze(1),train_yield,p,covar_module=kernel)
    fit_gpytorch_mll(mll_yield)
    with torch.no_grad():
        posterior_yield = model_yield.posterior(train_x_normalized[:,0].unsqueeze(1))
        noises.append(float(posterior_yield.variance.mean())**0.5)
        r2.append(1-float((posterior_yield.mean-train_yield.squeeze()).square().mean()))

print(lengthscales)
print(noises)
print(r2)
ax1.plot(lengthscales,noises)
ax2.plot(lengthscales,r2)
plt.show()