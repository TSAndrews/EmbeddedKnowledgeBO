import os
import torch
import time
from botorch.utils.transforms import unnormalize, normalize
from src.optimization_utils import generate_initial_sample, get_observation, optimize_qehvi, get_recomendations,objective_factory, initialize_model
import warnings
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
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

iterations=50
batch_size=1
mc_samples = 128
train_size=2#2 * (problem.dim + 1)

################################### Initialization ########################################################
train_x_raw, t0 = generate_initial_sample(problem,n=train_size)
train_obj_raw = get_observation(train_x_raw,problem)

################################### pre-modelling objective computation ####################################

train_x=train_x_raw.detach().clone()
train_obj= acquisition_objective.objective(train_obj_raw.detach().clone().unsqueeze(0),normalize(train_x,problem.bounds))


iter_time=[t0/train_size]*train_size
for _ in range(iterations):
    x_new,t=get_recomendations(train_x,train_obj,problem,model_initializer=initialize_model,acquisition_func=optimize_qehvi, batch_size=batch_size,mc_samples=mc_samples,model_initializer_kwargs={"noise":NOISE_SE},tkwargs=tkwargs)
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
    
train_x=train_x_raw.detach().clone()
train_obj=train_obj_raw.detach().clone()

iter_time=[t0/train_size]*train_size
for _ in range(iterations):
    x_new,t=get_recomendations(train_x,train_obj,problem,model_initializer=initialize_model,acquisition_func=optimize_qehvi, batch_size=batch_size,mc_samples=mc_samples,model_initializer_kwargs={"noise":NOISE_SE},acq_kwargs={"objective":acquisition_objective},tkwargs=tkwargs)
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
    
#-----------------------Save Data to csv-----------------------------
import pandas as pd
import numpy as np
xpre=train_x_pre.numpy()
xpost=train_x_post.numpy()
ypre=train_obj_pre.squeeze(0).numpy()
ypost=train_obj_post.squeeze(0).numpy()
tpre=np.asarray(iter_time_pre).reshape((-1,1))
tpost=np.asarray(iter_time_post).reshape((-1,1))
hpre=np.asarray(hypervolume_pre).reshape((-1,1))
hpost=np.asarray(hypervolume_post).reshape((-1,1))

data=pd.DataFrame(np.concatenate((xpre,xpost,ypre,ypost,tpre,tpost,hpre,hpost),axis=1),columns=["ResTimePre","equiv_pldnPre","conc_dfnbPre","temperaturePre","ResTimePost","equiv_pldnPost","conc_dfnbPost","temperaturePost","STYPre","EFactorPre","STYPost","EFactorPost","itterTimePre","itterTimePost","hypervolumePre","hypervolumePost"])
data.to_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)),"results","PreVsPostModellingData.csv"))
#---------------------------Plot-------------------------------------

plt.plot(hypervolume_pre)
plt.plot(hypervolume_post)
plt.show()
    
    
    
    
    
    
    


    






