# The Effects of Post-Modelling Performance Metric Computation on the Performance of Bayesian Optimizers

## Introduction
Bayesian optimization is often considered a black-box technique, however several simple modifications to 
the design can vastly increase the optimizers computational and iterative performance on certain tasks. The
modification explored here relates to what aspects of a problem are actually modeled during the optimization 
procedure. Raw data collected is not typically in a form that properly represents the value of the systems 
state and often needs to be transformed into one or more performance metrics that are desired to be optimized.
Traditionaly, these performance metrics are what get modeled during the Bayesian optimization process but this 
leads to a loss of the structural knowledge introduced by the transforms. If the paramters being optimized 
are featured in the performance calculations, knowledge of these influences are lost from the system. Modelling 
only the raw data may also reduce the number of models that need be fitted in multiobjective problems. There is 
no need to create a sepearate model for each if metrics if they are derived from the same raw data. This can 
result in significant computational savings for the optimization runs. Another potential benifit lies with the
form of the models themselves. Many performance metrics increase the complexity of the models. This increase in
complexity could degrade the fit of the models, and result in a decrease in the overall performance of the 
optimizer. The form of raw data moodels are also likely to be better understood by domain experts which could 
also increase the ease of designing covarience kernels that are better suited to the data. 

The impacts of performance metric computation order on computational and iterative effeciency were investigated using
the optimization of a simulated nucleophilic aromatic substitution reaction system as a case study. This simulated 
system was adapted from the summit software package<sup>1</sup> and is based on the works of Hone et 
al.<sup>2</sup>. The system was optimized to maximize space-time yield and minimize reaction E-factor. 
These are common industrial measures for production rate and material wastage respectively. Both of these
metrics can be calculated from the concentration of product produced by a reaction and the set experimental 
parameters such as duration and initial reactant quantities.

## Results
Computation of performance metrics from a single model for yield halved the computation time for each iteration of 
the optimizer compared to modelling both space-time yield and E-factor separately. This is to be expected. 
Model fitting is typically the most computationally intensive task during bayesian optimization and so 
reducing the number of models that need be fitted has a significant impact on computation time.

Post-modelling performance metric computation does not appear to have a significant impact on the convergence rate of 
the optimizer. When data has a low noise level, post-modelling performance metric computation does typically increase
hypervolume more rapidly than computing performance metrics prior to modelling in early iterations, but this lead drops 
off fairly quickly and both ojective computation methods generally reach the optima at aproximately the same
time. This is because post-modelling performance metric computation biases selection toward regions that have
high value experimental conditions based on the metrics without considering the responses of the system under these conditions.
For instance, the optimizer will be biased towards low residence times because time is a key component of space time yield, but
these conditions are not necessarily optimal because residence time and yields are inversly proportional. As such, the optima of
the system may not lie at minimal residence times if the drop in yields exceed the increase in speed. Post-modelling computation 
gets an early lead due to its better understanding of the impact of the impact of reactor paramters on the metrics values, but it
still needs to determine the effects of yields to determine the optima. The early bias likely hinders exploration of the domain
counteracting the benifits of increased knowledge integration as iterations progress. 



The drawbacks of the bias introduced by post-modelling metric computation could likely be counteracted by more intelligent design
of the priors of the optimizer. It is typically known that yields are likely to decrease with decreasing residence time which would
counteract the low residence time bias being introduced by the space time yield performance metric. The bias may even be useful in 
most systems by enabling domain reduction. There is a known upper bound for yields as we cannot create matter. This means that it 
can be known that increasing residence time cannot possibly improve metrics such as space time yield because yields may need to be
greater than possible in order to counteract the time effects. Biasing exploration exploration towards regions of short residence 
time increases the liklihood of being able to shrink the domain.

## Conclusions
Computing performance metrics after modelling can significantly reduce computation time if multiple metrics are derived from the same
measurments. These improvements will be especially noticable for problems with a large number of metrics to be optimized based on only 
a few different sensors.
Computing performance metrics after modelling increases the early rate of improvement of the system due to the incrreased knowledge of
the effects of reactor parameters on the metrics, but this early bias likely deacreases exploration rate, and thus does little to improve
the overall optimization times. Its possible that these drawbacks could be overcome by the use of non-constant priors or by mearly 
enforcing a spacing criteria during experimental selection in early iterations.

## References:
1. Felton, K. C., Rittig, J. G., & Lapkin, A. A. (2021), [Summit: benchmarking machine learning methods for reaction optimisation](https://doi.org/10.1002/cmtd.202000051). _Chemistry‐Methods_, _1_(2), 116-122.
2. Hone, C. A., Holmes, N., Akien, G. R., Bourne, R. A., & Muller, F. L. (2017). [Rapid multistep kinetic model generation from transient flow data](https://doi.org/10.1039/C6RE00109B). _Reaction chemistry & engineering_, _2_(2), 103-108.


