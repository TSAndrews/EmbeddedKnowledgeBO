#Parameter Exploration
##Post vs Pre modelling objective computation no noise
|File|Training Size|MCSamples|Iterations|
|----|-------------|---------|----------|
|PreVsPostModellingData1.csv|2|128|50|

##Post vs Pre modelling objective computation with noise
|File|Training Size|MCSamples|Iterations|
|----|-------------|---------|----------|
|PreVsPostModellingDataNoisy1.csv|5|128|50|

##Lengthscale bounds
|File|Min Lengthscale|Training Size|MCSamples|Iterations|Noise|
|----|---------------|-------------|---------|----------|-----|
|MinLengthscaleData1.csv|0.5|5|128|40|0.1|

##Adaptive Lengthscale bounds
|File|Lengthscale func|Training Size|MCSamples|Iterations|Noise|
|----|----------------|-------------|---------|----------|-----|
|MinLengthscaleDataAdaptiveNoise1.csv|(1/n)*5||5|50|0.01|