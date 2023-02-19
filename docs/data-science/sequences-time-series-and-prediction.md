# Coursera - sequences, time series, and prediction

## Sequences and Prediction

### machine learning applied to time series

- prediction or forecasting
- imputation/归因 归责：fillup discontinuous data
- anomaly detection
- pattern recognition

### common patterns in time series

- trend
- seasonality
- auto correlated

$$
v(t) = 0.99 × v(t-1)+occasional\ spike
$$

- noise
- stationary or non-stationary

### moving average

- averaging window
- differencing: remove the trend and seasonality
- trailing versus centered windows

差分是通过**计算相邻观测值之间的差值**让非平稳时间序列变平稳的方法。差分可以通过去除时间序列中的一些变化特征来平稳化它的均值，并因此消除（或减小）时间序列的趋势和季节性。

自相关图（ACF图）也能帮助我们识别非平稳时间序列。 对于一个平稳时间序列，自相关系数（ACF）会快速的下降到接近 0 的水平，然而非平稳时间序列的自相关系数会下降的比较缓慢。

