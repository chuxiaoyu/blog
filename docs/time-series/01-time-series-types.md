# Taxonomy of Time Series Forecasting
### Inputs vs. Outputs
Inputs: Historical data provided to the model in order to make a single forecast.

Outputs: Prediction or forecast for a future time step beyond the data provided as input.
### Endogenous vs. Exogenous
Endogenous: Input variables that are influenced by other variables in the system and on which the output variable depends. For example, the observation at time t is dependent upon the observation at t−1; t−1 may depend on t-2, and so on.

Exogenous: Input variables that are not influenced by other variables in the system and on which the output variable depends.

### Regression vs. Classification
Regression: Forecast a numerical quantity.

Classification: Classify as one of two or more labels.

*A regression problem can be reframed as classification and a classification problem can be reframed as regression.*

### Unstructured vs. Structured
Unstructured: No obvious systematic time-dependent pattern in a time series variable.

Structured: Systematic time-dependent patterns in a time series variable (e.g. trend and/or seasonality).

### Univariate vs. Multivariate
Univariate: One variable measured over time.
```
time, measure 
1, 100 
2, 110 
3, 108 
4, 115 
5, 120
```

Multivariate: Multiple variables measured over time
```
time, measure1, measure2 
1, 0.2, 88 
2, 0.5, 89 
3, 0.7, 87 
4, 0.4, 88 
5, 1.0, 9
```

*Considering this question with regard to inputs and outputs may add a further distinction.*

Univariate and Multivariate Inputs: One or multiple input variables measured over time.

Univariate and Multivariate Outputs: One or multiple output variables to be predicted.

### Single-step vs. Multi-step

One-step: Forecast the next time step.

Multi-step: Forecast more than one future time steps.

### Static vs. Dynamic
Static: A forecast model is fit once and used to make predictions.

Dynamic: A forecast model is fit on newly available data prior to each prediction.

### Contiguous vs. Discontiguous
Contiguous: Observations are made uniform over time.

Discontiguous: Observations are not uniform over time.