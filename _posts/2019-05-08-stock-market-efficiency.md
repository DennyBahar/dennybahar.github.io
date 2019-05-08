---
title: Stock Market Efficiency
date: 2019-05-08
tags: ["finance"]
excerpt: "Exploring the dynamic of US stock market efficiency."
---


The Efficient Market Hypothesis (EMH) states all available information is already priced in market. This implies that past performance does not hold any clue as to what the future perfomance of the market will be.
As a result, the market move randomly and there is no way to forecast its future performance.

This article aim to explore efficiency of the US stock market. The S&P 500 monthly data from year 1871 to 2018 is used. The source of the data is from Robert Shiller and retrieved from <a href="https://datahub.io/core/s-and-p-500">here</a>

```python
# read dataset
sp500 = pd.read_csv("sp500.csv", index_col='Date', parse_dates=True)

```

{{sp500.info()}}

To measure the market efficiency, I calculate the 60 months (5 years) autocorrelation for this month return and previous month return. The interpretations of the autocorrelation are as follow:

1. <b>Zero correlation</b> means future returns have <u>no relationship</u> with past returns. Thus, Market is efficient.
2. <b>Positive correlation</b> means future returns tend to move in the <u>same direction</u> as past returns. If the stock market increase in the previous month, then the stock market will more likely rise in the next month as well and vice versa.
3. <b>Negative correlation</b> means futures returns tend to move in the <u>opposite direction</u> as past returns. If the stock market increase in the previous month, then the stock market will more likely to fall in the next month and vice versa.  

```python
# monthly return
sp500['return'] = sp500['SP500'].pct_change()
# previous month return
sp500['return lag'] = sp500['return'].shift()
# 60 months correlation
sp500['rolling corr'] = sp500['return'].rolling(60).corr(sp500['return lag'])
```

```python
# plotting
fig, ax = plt.subplots(nrows=2,figsize=(12,10))

ax[0].plot(sp500['SP500'])
ax[0].set_title("S&P 500 Price (1871-2018)",size='large')
ax[0].set_ylabel('Price USD')
ax[0].set_xlabel('Date')

ax[1].plot(sp500['rolling corr'])
ax[1].set_title("S&P 500 60-Months Rolling Autocorrelation (1871-2018)", size='large')
ax[1].set_ylabel('Autocorrelation')
ax[1].set_xlabel('Date')

fig.tight_layout()
```