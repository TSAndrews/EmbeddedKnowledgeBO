# Embedding System Knowledge into Bayesian Optimizers

## Introduction
Bayesian optimization is often considered a black-box technique, however several simple modifications to 
the design can vastly increase the optimizers computational and iterative performance on certain tasks. 
Two such modifications will be explored in this work; post-modelling objective computation for enhanced 
computational efficiency and improved knowledge integration, and minimum length-scale enforcement for 
increased stability of optimizers to system noise when using homoscedastic noise modelling.

Post-modelling objective computation often reduces the number of models that need be fitted to the data 
and can result in them having a more representative posterior as the transformations created by the 
formulating the objectives can be decoupled from the raw measurement values.

Bayesian optimizers utilizing homoscedastic noise modelling must balance both fitting of both the kernels 
and the noise. Methods optimizing the marginal likelihood are prone to overfitting the kernels during 
early iterations leading to suboptimal algorithm performance. This is due to the presence of a second 
optima in the marginal likelihood curve with respect to length-scale which occurs when the model stops
explaining the system noise with the fitted kernels and instead begins correctly capturing it in the 
inferred system noise. Providing a minimum length-scale can help locate this secondary optima and ensure
that the kernels only capture the base trends and not the system noise. 

The impacts of these modifications on computational and iterative performance will be demonstrated using
the optimization of a simulated nucleophilic aromatic substitution reaction system. This simulated 
system was adapted from the summit software package<sup>1</sup> and is based on the works of Hone et 
al.<sup>2</sup>.

## Results
### The effect of post-modelling objective computation
Computation of objectives after the fitting of models results in a significant reduction in the required 
computation cost when identifying exploration points for the nucleophilic aromatic substitution system.
This is due to the fact that only one model need be fitted to describe both objectives as both space-time 
yield and E-factor can be computed from the measured yields of the system. Bayesian optimization 
iterations were typically twice as fast when computing objectives after modelling due to this reduction 
in dimensionality during the modelling process. Computation cost generally decreases over time, likely due to 
the increased ease of optimizing kernel hyperparameters to maximize the likelihood  when more data is present.

Post-modelling objective computation does not appear to have a significant impact on the convergence rate of 
the optimizer. When data has a low noise level, post-modelling objective computation does typically increase
hypervolume more rapidly than pre-modelling objective computation in early iterations, but this lead drops 
off fairly quickly and both ojective computation methods generally reach the optima in aproximately the same
number of intervals. The post-modelling objective computation effectively acts as a non constant prior in 
these examples, and so the early lead is likely a result of this bias pushing selection to regions that have
more potential. The early lead is lost because the true optima does not lie at the optima of the prior, 
however, since the post-modelling system preferencially targets experiments with higher theoretical maxima 
based on the objectives, it may be possible to terminate the optimization process earlier if the upper bound
of possible system response is also considered during the selection cycle. The addition of noise to the 
system removes the early lead in hypervolume improventfor post-modelling objective computation over 
pre-modelling computation. Hypervolume ncreases at aproximately the same rte in such cases. This is likely 
due to the increased difficulty of resolving system noise in early iterations due to the low densiy of 
experimentation.

### The impacts of enforcing minimal lengthscales on noise stability

## Conclusions

## References:
1. Felton, K. C., Rittig, J. G., & Lapkin, A. A. (2021), [Summit: benchmarking machine learning methods for reaction optimisation](https://doi.org/10.1002/cmtd.202000051). _Chemistry‐Methods_, _1_(2), 116-122.
2. Hone, C. A., Holmes, N., Akien, G. R., Bourne, R. A., & Muller, F. L. (2017). [Rapid multistep kinetic model generation from transient flow data](https://doi.org/10.1039/C6RE00109B). _Reaction chemistry & engineering_, _2_(2), 103-108.


