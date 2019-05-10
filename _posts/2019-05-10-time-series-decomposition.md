---
title: "Time Series Decomposition"
date: 2019-05-10
tags: [time series]
excerpt: decompose time series into trend, seasonal and noise components
mathjax: true
---

Often times time series data have trends or seasonality or both. Hence, it is advantageous to seperate the components to have more clarity in the analysis and improve to forecasting. The time series $$y_t$$ can be thought to have three components: trend ($$T_t$$), seasonal ($$S_t$$) and noise ($$\epsilon_t$$). Using these components, the series can be modelled in two ways:
* <b>Additive:</b> $$y_t = T_t + S_t + \epsilon_t$$
* <b>Multiplicative:</b> $$y_t = T_t \times S_t \times \epsilon_t$$

Additive model is suitable for time series which trend and seasonality seems to be linear. Multiplicative model is suitable for time series which trend and seasonality seems to be non-linear (growing/falling over time).

The data to be used is the classic air passengers monthly data from year 1949 to 1960. The number of passengers is in thousands.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
```


```python
df = pd.read_csv('dataset/AirPassengers.csv',index_col='Month',parse_dates=True)
print(df.info())
print(df.head())
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 144 entries, 1949-01-01 to 1960-12-01
    Data columns (total 1 columns):
    #Passengers    144 non-null int64
    dtypes: int64(1)
    memory usage: 2.2 KB
    None
                #Passengers
    Month                  
    1949-01-01          112
    1949-02-01          118
    1949-03-01          132
    1949-04-01          129
    1949-05-01          121
    

```python
df.plot(figsize=(10,5))
```


![png](/images/time-series-decomposition_files/time-series-decomposition_2_1.png)

From the image above, it is clear that the series exhibit an upward trend and there is seasonal pattern. The number of passengers peaked during summer and winter period (holiday seasons). Moreover, the trend and sesonality seems to grow non-linearly. Therefore, multiplicative model will be used. 


```python
res = seasonal_decompose(df, model='multiplicative') # res.trend, res.seasonal, res.resid
res.plot();
```


![png](/images/time-series-decomposition_files/time-series-decomposition_3_0.png)

After decompose the time series into the three components, estimating the growth rate of the passengers and the period of the season will be straightforward.