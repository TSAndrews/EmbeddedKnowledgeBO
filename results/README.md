# Parameter Exploration
## Post vs Pre modelling objective computation no noise
|File|Training Size|MCSamples|Iterations|
|----|-------------|---------|----------|
|PreVsPostModellingData1.csv|2|128|50|

## Post vs Pre modelling objective computation with noise
|File|Training Size|MCSamples|Iterations|Noise|
|----|-------------|---------|----------|-----|
|PreVsPostModellingDataNoisy1.csv|5|128|50|0.1|
|PreVsPostModellingDataNoisy2.csv|5|128|50|0.01|

## Lengthscale bounds
|File|Min Lengthscale|Training Size|MCSamples|Iterations|Noise|
|----|---------------|-------------|---------|----------|-----|
|MinLengthscaleData1.csv|0.5|5|128|50|0.1|
|MinLengthscaleData2.csv|0.5|5|128|50|0.01|


## Adaptive Lengthscale bounds
|File|Lengthscale func|Training Size|MCSamples|Iterations|Noise|
|----|----------------|-------------|---------|----------|-----|
|MinLengthscaleDataAdaptiveNoise1.csv|(1/n)*5||5|50|0.01|
|MinLengthscaleDataAdaptiveNoise2.csv|(1/n)*5||5|50|0.1|
|MinLengthscaleDataAdaptiveNoise3.csv|(1/n)*3||5|50|0.1|
|MinLengthscaleDataAdaptiveNoise4.csv|(1/n)*3||5|50|0.01|