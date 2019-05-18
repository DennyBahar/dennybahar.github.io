---
title: Holt-Winter Method
date: 2019-05-11
tags: [time series]
excerpt: Holt-Winter method of time series forecasting
mathjax: true
---

Holt-Winter is a forecasting method which takes into account the trend and seasonal components of the series. Trend is also known as "gradient" or "rate of change" and seasonal is a repeating periodic fluctuations. Thus, this method is inapproriate for system which exhibit non-periodic fluctuations (e.g. stock market).

This article describe how the Holt-Winter method makes forecast on time series data. The data to be used is the classic air passengers dataset from year 1949 to 1960.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    

It is good practice to set the frequency of the time series data because time series analysis depend on the context of the time steps.


```python
# assign frequency to be monthly
df.index.freq = 'MS'
print(df.index)
```

    DatetimeIndex(['1949-01-01', '1949-02-01', '1949-03-01', '1949-04-01',
                   '1949-05-01', '1949-06-01', '1949-07-01', '1949-08-01',
                   '1949-09-01', '1949-10-01',
                   ...
                   '1960-03-01', '1960-04-01', '1960-05-01', '1960-06-01',
                   '1960-07-01', '1960-08-01', '1960-09-01', '1960-10-01',
                   '1960-11-01', '1960-12-01'],
                  dtype='datetime64[ns]', name='Month', length=144, freq='MS')
    


```python
df.plot(figsize=(12,6))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1d338ebe400>




![png](/images/holt-winters-method_files/holt-winters-method_3_1.png)


To evaluate the forecasting method, the dataset is split into training set (1949 to 1957) and test set (1958 to 1960). The mean squared error (MSE) will be use as the criteria of the evaluation.

<p style="text-align:center;">$$MSE = \frac 1T \sum_{t=1}^T (y_t - \hat y_t)^2$$</p>

  
```python
train_period = ("1949-01-01","1957-12-01")
test_period = ("1958-01-01",'1960-12-01')

# seperate into training and test set
train_df = df.loc[train_period[0]:train_period[1]].copy()
test_df = df.loc[test_period[0]:test_period[1]].copy()

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(train_df, color='blue',label='training set')
ax.plot(test_df, color='orange',label='test set')
ax.legend()
```




    <matplotlib.legend.Legend at 0x1d339848f28>




![png](/images/holt-winters-method_files/holt-winters-method_4_1.png)



```python
# to calculate mean squared error (MSE)
def mse(df):
    arr = df.values
    sse = np.sum((arr[:,1] - arr[:,0])**2)
    n = len(arr)
    
    return sse/n


# to visualize
def show_visual(train_df, test_df, title):
    fig, ax = plt.subplots(figsize=(12,6))
    
    ax.plot(train_df.iloc[:,0], color='blue', label='actual training set')
    ax.plot(train_df.iloc[:,1], color='red', label='prediction training set', linestyle="--")
    
    ax.plot(test_df.iloc[:,0], color='orange', label='actual test set')
    ax.plot(test_df.iloc[:,1], color='green', label='prediction test set', linestyle="--")
    
    ax.set_title("Prediction Method: "+title)
    ax.set_xlabel('Month')
    ax.set_ylabel('#Passengers (thousands)')
    
    ax.legend()
```

# Single Exponential Smoothing

Single exponential smoothing (SES) is the first principle in which Holt-Winter method is based on. The $$y_t$$ is the <u>most recent information</u> and the $$\hat y_{t-1}$$ is a recursive which contain all the past values <u>(memory)</u>. Thus, this method can be thought as a type of weighted average where the weight decreases exponentially with past values. The smoothing factor $$\alpha$$ control how fast the weight decreases and it has a range $$0 \le \alpha \le 1$$. Larger $$\alpha$$ assign more weights to recent values. SES is not a forecasting method, but more of a method to describe the observed data. As a result, SES will perform naive forecasting if it is use forcasting.


<p style="text-align:center;">$$SES:\;\hat y_{t} = \alpha \cdot y_t + (1-\alpha)\cdot \hat y_{t-1}$$</p>
<p style="text-align:center;">$$Naive\; Forecast:\;\hat y_{t+1} = y_t$$</p>

```python
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# fit model and predict on test set
model1 = SimpleExpSmoothing(train_df['#Passengers']).fit(optimized=True)
train_df['SES'] = model1.fittedvalues.values
test_df['SES'] = model1.predict(test_period[0],test_period[1]).values

# caculate MSE
print("Training set MSE: {:.2f}".format(mse(train_df[['#Passengers','SES']])))
print("Test set MSE: {:.2f}".format(mse(test_df[['#Passengers','SES']])))

# plot
show_visual(train_df[['#Passengers','SES']], test_df[['#Passengers','SES']], 'Single Exponential Smoothing')
```

    Training set MSE: 661.31
    Test set MSE: 14674.56
    


![png](/images/holt-winters-method_files/holt-winters-method_7_1.png)


# Double Exponential Smoothing (Holt Method)

Holt method is a forecasting method which consider the trend component to make prediction. This method perform exponential smoothing on the level values ($$l_t$$ is the expected $$y_t$$ or $$\hat y_t$$) and the trend values ($$b_t$$). A total of two exponential smoothing. Hence, it require two smoothing factors: $$\alpha$$ for the level component and $$\beta$$ for the trend component. The Holt method forecast $$\hat y_{t+m}$$ resemble a straight line equation ($$y=mx+c$$) where level ($$l_t$$) is the estimated intercept and trend ($$b_t$$) is the estimated rate of change at time $$t$$. 

<p style="text-align:center;">$$Level:\; l_t = \alpha y_t + (1-\alpha)(l_{t-1} + b_{t-1})$$</p>
<p style="text-align:center;">$$Trend:\; b_t = \beta (l_t - l_{t-1}) + (1-\beta)b_{t-1}$$</p>
<p style="text-align:center;">$$Forecast:\; \hat y_{t+m} = l_t + mb_t$$</p>


```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# fit model and predict on test set
model2 = ExponentialSmoothing(train_df['#Passengers'], trend='mul').fit(optimized=True)
train_df['holt'] = model2.fittedvalues.values
test_df['holt'] = model2.predict(test_period[0],test_period[1]).values

# caculate MSE
print("Training set MSE: {:.2f}".format(mse(train_df[['#Passengers','holt']])))
print("Test set MSE: {:.2f}".format(mse(test_df[['#Passengers','holt']])))

# plot
show_visual(train_df[['#Passengers','holt']], test_df[['#Passengers','holt']], 'Holt Method')
```

    Training set MSE: 661.01
    Test set MSE: 11622.11
    


![png](/images/holt-winters-method_files/holt-winters-method_9_1.png)


# Triple Exponential Smoothing (Holt-Winter Method)

The Holt-Winter method make prediction by considering the trend and seasonal components. It perform exponential smoothing on the level values ($$l_t$$), trend ($$b_t$$) and seasonality ($$s_t$$). A total of three exponential smoothing. Hence, it require three smoothing factors: $$\alpha$$ for the level component and $$\beta$$ for the trend component and $$\gamma$$ for the seasonal component. The Holt-Winter method also require the length of the season ($$L$$) as an input. The index of $$s_{t-L+1+(m-1)modL}$$ is to offset the seasonal value such that it only use values from the past observed values and not the future predicted values.

<p style="text-align:center;">$$Level:\; l_t = \alpha y_t + (1-\alpha)(l_{t-1} + b_{t-1})$$</p>
<p style="text-align:center;">$$Trend:\; b_t = \beta (l_t - l_{t-1}) + (1-\beta)b_{t-1}$$</p>
<p style="text-align:center;">$$Seasonal:\; s_t = \gamma(y_t - l_t) + (1-\gamma)s_{t-L}$$</p>
<p style="text-align:center;">$$Forecast:\; \hat y_{t+m} = l_t + mb_t + s_{t-L+1+(m-1)modL}$$</p>


```python
# fit model and predict on test set
model3 = ExponentialSmoothing(train_df['#Passengers'], trend='mul', seasonal='mul', seasonal_periods=12).fit(optimized=True)
train_df['holt'] = model3.fittedvalues.values
test_df['holt'] = model3.predict(test_period[0],test_period[1]).values

# caculate MSE
print("Training set MSE: {:.2f}".format(mse(train_df[['#Passengers','holt']])))
print("Test set MSE: {:.2f}".format(mse(test_df[['#Passengers','holt']])))

# plot
show_visual(train_df[['#Passengers','holt']], test_df[['#Passengers','holt']], 'Holt-Winter Method')
```

    Training set MSE: 79.08
    Test set MSE: 376.02
    


![png](/images/holt-winters-method_files/holt-winters-method_11_1.png)


In this example, the smoothing factors selected are based on the optimized values found on the training set. However, sometime not optimizing the smoothing factors based on the training set may produce better result because the future in uncertain (bias-variance tradeoff). Furthermore, series don't usually increase/decrease infinitely. At some point of time it will start to slow down. In that case, adding a <b>damping factor $$(\phi)$$</b> is usually advisable.