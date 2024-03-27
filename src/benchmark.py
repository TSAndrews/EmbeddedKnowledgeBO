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
        
    def evaluate_true(self, X: Tensor) -> Tensor:
        r"""Evaluate the function (w/o observation noise) on a set of points."""
        Y=torch.empty((0,1))
        for tau,equiv_pldn,conc_dfnb,temperature in X:
            C_i = np.zeros(5)
            C_i[0] = conc_dfnb
            C_i[1] = equiv_pldn * conc_dfnb
            res = solve_ivp(self._integrand, [0, tau], C_i, args=(temperature,),kwargs={"C_i":C_i})
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
