from gpytorch.priors.torch_priors import GammaPrior
distribution=GammaPrior(3,6)
print(distribution.mean,distribution.variance)