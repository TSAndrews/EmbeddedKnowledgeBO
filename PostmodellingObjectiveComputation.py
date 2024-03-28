import os
import torch
import time
from botorch.utils.transforms import unnormalize, normalize
from src.optimization_utils import generate_initial_sample, get_observation, optimize_qehvi, get_recomendations,GenericMCMultiObjective
import warnings
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize
#from botorch.utils.multi_objective.pareto import is_non_dominated


warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")
NOISE_SE = torch.tensor([1e-3]*4, **tkwargs)

# ### Problem setup
# from botorch.test_functions.multi_objective import BraninCurrin
# problem = BraninCurrin(negate=True).to(**tkwargs)
from src.benchmark import SnArModel
problem = SnArModel().to(**tkwargs)

def initialize_model_fixed_noise(train_x, train_obj,problem,noise):
    # define models for objective and constraint
    train_x = normalize(train_x, problem.bounds)
    models = []
    for i in range(train_obj.shape[-1]):
        train_y = train_obj[..., i : i + 1]
        train_yvar = torch.full_like(train_y, noise[i] ** 2)
        models.append(SingleTaskGP(train_x, train_y, train_yvar, outcome_transform=Standardize(m=1)))
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model

def compute_space_time_yield(Y,X):
    if Y.dim()-X.dim()==1:
        X=X.unsqueeze(0)
    res_time=X[...,0].unsqueeze(-1)
    sty=torch.div(Y,res_time)
    return sty

def compute_e_factor(Y,X):
    if Y.dim()-X.dim()==1:
        X=X.unsqueeze(0)
    equiv_pldn=X[...,1].unsqueeze(-1)
    conc_dfnb=X[...,2].unsqueeze(-1)
    
    rho_eth = 789  # g/L (should adjust to temp, but just using @ 25C)
    
    mass_reagents=(rho_eth + conc_dfnb*159.09 + equiv_pldn*conc_dfnb*71.11)#g/L. *1e3*q_tot = g/min 
    mass_products=(Y*210.21)#g/L. *1e3*q_tot = g/min
    mass_products[mass_products<1e-5]=1e-5
    e_factor = mass_reagents / mass_products
    e_factor[e_factor>1e3]=1e3
    return 1e3-e_factor #minus to convert to maximization, 1e3 so ref_point can be saftely set to 0,0

def objective_factory(objectives):
    """"""
    def objective(Y,X):
        X_raw=unnormalize(X,problem.bounds)
        Ynew=[obj(Y,X_raw) for obj in objectives]
        return torch.cat(Ynew,dim=-1)
    return GenericMCMultiObjective(objective)

acquisition_objective=objective_factory([compute_space_time_yield,compute_e_factor])

iterations=25
batch_size=1
mc_samples = 128
train_size=2 * (problem.dim + 1)

################################### pre-modelling objective computation ####################################

# call helper functions to generate initial training data and initialize model
train_x, t0 = generate_initial_sample(problem,n=train_size)
train_obj_raw = get_observation(train_x,problem)
train_obj = acquisition_objective.objective(train_obj_raw.unsqueeze(0),normalize(train_x,problem.bounds))

iter_time=[t0/train_size]*train_size
for _ in range(iterations):
    x_new,t=get_recomendations(train_x,train_obj,problem,model_initializer=initialize_model_fixed_noise,acquisition_func=optimize_qehvi, batch_size=batch_size,mc_samples=mc_samples,model_initializer_kwargs={"noise":NOISE_SE})
    obj_new_raw=get_observation(x_new,problem)
    obj_new = acquisition_objective.objective(obj_new_raw.unsqueeze(0),normalize(x_new,problem.bounds))
    
    train_x=torch.cat([train_x,x_new])
    train_obj=torch.cat([train_obj,obj_new],dim=-2)
    iter_time.append(t)

train_x_pre=train_x.detach().clone()
train_obj_pre=train_obj.squeeze(dim=1).detach().clone()
iter_time_pre=iter_time.copy()
print("pre modelling test done!")
################################### post-modelling objective computation ####################################
    
# call helper functions to generate initial training data and initialize model
train_x, t0 = generate_initial_sample(problem,n=train_size)
train_obj = get_observation(train_x,problem)

iter_time=[t0/train_size]*train_size
for _ in range(iterations):
    x_new,t=get_recomendations(train_x,train_obj,problem,model_initializer=initialize_model_fixed_noise,acquisition_func=optimize_qehvi, batch_size=batch_size,mc_samples=mc_samples,model_initializer_kwargs={"noise":NOISE_SE},acq_kwargs={"objective":acquisition_objective},tkwargs=tkwargs)
    obj_new=get_observation(x_new,problem)
    
    train_x=torch.cat([train_x,x_new])
    train_obj=torch.cat([train_obj,obj_new],dim=-2)
    iter_time.append(t)

train_x_post=train_x.detach().clone()
train_obj_post=acquisition_objective.objective(train_obj.unsqueeze(0),normalize(train_x,problem.bounds)).squeeze(dim=1).detach().clone()
iter_time_post=iter_time.copy()
print("post modelling test done!")
################################### plots & analysis ########################################################
import matplotlib.pyplot as plt
#----------------------------------Pre-------------------------------
hypervolume_pre=[]
for i in range(train_x_pre.shape[0]):
    bd = DominatedPartitioning(ref_point=problem.ref_point, Y=acquisition_objective.objective(problem(train_x_pre[0:i]).unsqueeze(0),normalize(train_x_pre[0:i],problem.bounds)))
    hypervolume_pre.append(bd.compute_hypervolume().item())

#----------------------------------Post------------------------------
hypervolume_post=[]
for i in range(train_x_post.shape[0]):
    bd = DominatedPartitioning(ref_point=problem.ref_point, Y=acquisition_objective.objective(problem(train_x_post[0:i]).unsqueeze(0),normalize(train_x_post[0:i],problem.bounds)))
    hypervolume_post.append(bd.compute_hypervolume().item())

plt.plot(hypervolume_pre)
plt.plot(hypervolume_post)
plt.show()
    
    
    
    
    
    
    


    






