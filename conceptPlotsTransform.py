import os
import torch
import time
from botorch.utils.transforms import unnormalize, normalize
from src.optimization_utils import generate_initial_sample,objective_factory, initialize_model
from botorch import fit_gpytorch_mll
import warnings
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
import matplotlib.pyplot as plt
from collections import namedtuple
#from botorch.utils.multi_objective.pareto import is_non_dominated


warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")
NOISE_SE = torch.tensor([1e-3]*2, **tkwargs)

# ### Problem setup
# from botorch.test_functions.multi_objective import BraninCurrin
# problem = BraninCurrin(negate=True).to(**tkwargs)
from src.benchmark import SnArModel, compute_space_time_yield,compute_e_factor
problem = SnArModel().to(**tkwargs)

acquisition_objective=objective_factory([compute_space_time_yield,compute_e_factor],problem)


#Space time yield posteria against residence time: pre vs post objective computation
from botorch.test_functions.base import BaseTestProblem
class Dummy(BaseTestProblem):
    _bounds=[problem._bounds[0]]
    dim=1
    def evaluate_true(self, X):
        return X[...,0]
p=Dummy().to(**tkwargs)
n_train=10
n_test=100
f, (ax1,ax2) = plt.subplots(1, 2, figsize=(6, 4))

torch.manual_seed(1)
train_x_normalized=0.5*torch.ones((n_train,4))
train_x_normalized[:,0]=torch.rand((n_train))
train_x=unnormalize(train_x_normalized,problem.bounds)
train_yield=problem(train_x)
train_sty=compute_space_time_yield(train_yield,train_x)

mll_yield,model_yield=initialize_model(train_x[:,0].unsqueeze(1),train_yield,p)
mll_sty,model_sty=initialize_model(train_x[:,0].unsqueeze(1),train_sty,p)

fit_gpytorch_mll(mll_yield)
fit_gpytorch_mll(mll_sty)

test_X= torch.linspace(p._bounds[0][0], p._bounds[0][1], n_test, **tkwargs).unsqueeze(1)

with torch.no_grad():
    posterior_yield = model_yield.posterior(normalize(test_X,p.bounds))
    posterior_sty = model_sty.posterior(normalize(test_X,p.bounds))
    
    lower_yield, upper_yield = posterior_yield.mvn.confidence_region()
    mean_yield=posterior_yield.mean
    lower_sty, upper_sty = posterior_sty.mvn.confidence_region()
    mean_sty = posterior_sty.mean
    
    lower_yield_sty = compute_space_time_yield(lower_yield.unsqueeze(1),test_X).squeeze()
    upper_yield_sty = compute_space_time_yield(upper_yield.unsqueeze(1),test_X).squeeze()
    mean_yield_sty = compute_space_time_yield(mean_yield,test_X).squeeze()

    ax1.fill_between(test_X[:,0].cpu().numpy(), lower_sty.cpu().numpy(), upper_sty.cpu().numpy(), alpha=0.5)
    ax2.fill_between(test_X[:,0].cpu().numpy(), lower_yield_sty.cpu().numpy(), upper_yield_sty.cpu().numpy(), alpha=0.5)
    
    ax1.plot(test_X[:,0].cpu().numpy(), mean_sty.cpu().numpy(), 'b')
    ax2.plot(test_X[:,0].cpu().numpy(), mean_yield_sty.cpu().numpy(), 'b')
    
    ax1.plot(train_x[:,0].cpu().numpy(), train_sty.cpu().numpy(), 'k*')
    ax2.plot(train_x[:,0].cpu().numpy(), train_sty.cpu().numpy(), 'k*')
    
    ax1.set_ylim((0.04,0.05))
    ax2.set_ylim((0.04,0.05))

plt.show()