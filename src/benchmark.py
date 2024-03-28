import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch import Tensor
from botorch.test_functions.base import BaseTestProblem


class SnArModel(BaseTestProblem):
    """X=tau,equiv_pldn,conc_dfnb,temperature, Y=conc_products"""
    dim=4
    num_objectives=1
    _bounds=[(0.5,2.0),(1.0,5.0),(0.1,0.5),(30,120)]
    _ref_point = [0, 0]
    ref_point = torch.tensor(_ref_point, dtype=torch.float)
    
        
    def evaluate_true(self, X: Tensor) -> Tensor:
        r"""Evaluate the function (w/o observation noise) on a set of points."""
        Y=torch.empty((0,1))
        for tau,equiv_pldn,conc_dfnb,temperature in X:
            C_i = np.zeros(5)
            C_i[0] = conc_dfnb
            C_i[1] = equiv_pldn * conc_dfnb
            res = solve_ivp(self._integrand, [0, tau], C_i, args=(temperature,C_i))
            C_final = res.y[:, -1]
            conc_product=C_final[-1].reshape((-1,1))
            Y=torch.cat([Y,torch.from_numpy(conc_product)],dim=0)
        return Y
        
    def _integrand(self, t, C, T, C_i):
        # Kinetic Constants
        R = 8.314 / 1000  # kJ/K/mol
        T_ref = 90 + 273.71  # Convert to deg K
        T = T + 273.71  # Convert to deg K
        # Need to convert from 10^-2 M^-1s^-1 to M^-1min^-1
        k = (
            lambda k_ref, E_a, temp: 0.6
            * k_ref
            * np.exp(-E_a / R * (1 / temp - 1 / T_ref))
        )
        k_a = k(57.9, 33.3, T)
        k_b = k(2.70, 35.3, T)
        k_c = k(0.865, 38.9, T)
        k_d = k(1.63, 44.8, T)

        # Reaction Rates
        r = np.zeros(5)
        for i in [0, 1]:  # Set to reactants when close
            C[i] = 0 if C[i] < 1e-6 * C_i[i] else C[i]
        r[0] = -(k_a + k_b) * C[0] * C[1]
        r[1] = -(k_a + k_b) * C[0] * C[1] - k_c * C[1] * C[2] - k_d * C[1] * C[3]
        r[2] = k_a * C[0] * C[1] - k_c * C[1] * C[2]
        r[3] = k_b * C[0] * C[1] - k_d * C[1] * C[3]
        r[4] = k_c * C[1] * C[2] + k_d * C[1] * C[3]
        
        return r
    
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

