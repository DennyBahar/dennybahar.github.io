---
title: "Stock Market Returns Distribution"
date: 2019-05-09
tags: [finance, statistic]
excerpt: Explore the distribution of US stock market return.
mathjax: true
---

The normal distribution is most commonly used to model the returns of stock market. However, the market is well-known to exhibit rare disastrous event (black-swan event). To incorporate this into the model, fat-tail distribution is use.

There are various fat-tail distribution such as, student's t-distribution, pareto distribution, exponential distribution and many more. This article will explore the normal and t-distribution. The S&P 500 monthly data from year 1871 to 2018 is used. The source of the data is from Robert Shiller and retrieved from <a href="https://datahub.io/core/s-and-p-500">here</a>


```python
sp500['return'] = sp500['SP500'].pct_change()
ret_arr = sp500['return'].values
# remove nan
ret_arr = ret_arr[~np.isnan(ret_arr)]

mu = np.mean(ret_arr)
sigma = np.std(ret_arr)

print(ret_arr)
print("Average monthly return: {}".format(mu))
print("Standard deviation of monthly return: {}".format(sigma))
```

    [ 0.01351351  0.02444444  0.02819957 ... -0.03033909 -0.0008835
     -0.02241404]
    Average monthly return: 0.004442307774714635
    Standard deviation of monthly return: 0.0405779529005325
    


```python
num_bins = 300

x = np.linspace(np.min(ret_arr), np.max(ret_arr), num_bins)
normal_dist = stats.norm.pdf(x, mu, sigma)
t_dist = stats.t.pdf(x, 1, mu, sigma)
```


```python
fig,ax = plt.subplots(nrows=2, ncols=1, figsize=(12,12))

ax[0].hist(ret_arr, bins=x, density=True)
ax[0].set_title('SP500 monthly return density', size='large')

ax[1].plot(x, normal_dist, label='normal distribution')
ax[1].plot(x, t_dist, label='t-distribution')
ax[1].set_title("Normal distribution pdf & Student's t-distribution pdf (df = 1)",size='large')
ax[1].legend()

plt.tight_layout()
```


![png](/images/stock-return-dist_files/stock-return-dist_4_0.png)


The normal distribution takes in mean and standard deviation as inputs $$r \sim N(\mu, \sigma)$$. The t-distribution takes in mean, standard deviation and degree of freedom (df) as inputs $$r \sim T(\mu, \sigma, \nu)$$. The t-distribution above use df = 1. It is obvious that the t-distribution has fatter tail than the normal distribution. As df increases, the tail of the t-distribution gets skinnier and eventually equal to the normal distribution when df approaches infinity.


<p style="text-align:center;"> $$\lim_{\nu \to \infty} T(\mu, \sigma, \nu) = N(\mu, \sigma)$$ </p>

```python
fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))

ax.hist(ret_arr, bins=x, density=True, label='S&P 500 returns')
ax.plot(x, normal_dist, label='normal distribution')
ax.plot(x, t_dist, label='t-distribution')

ax.set_title('Comparison with normal and t-distribution', size='large')
ax.axvline(x=mu+2.5*sigma,color='r', linestyle='--', label='$\mu$ +- 2.5*$\sigma$')
ax.axvline(x=mu-2.5*sigma,color='r', linestyle='--')
ax.legend()

plt.tight_layout()
```


![png](/images/stock-return-dist_files/stock-return-dist_5_0.png)


Most of the middle portion of the stock market returns can be describe using normal distribution while the tails are better described using the t-distribution. Hence, in most cases (about 98% of the time) the stock market returns can be modelled using normal distribution while 2% of the time the stock market is better modelled using t-distribution. Though 2% seems like a tiny probability, with large sample that 1% probability will definitely occur and when it happen, it may harm (depending on the position direction) the unprepared. Therefore, it is prudent to not be mistaken between low probability with impossibility.

Thus, there is no perfect way to model the stock market returns because:
- Model using fat-tail distribution will incur lots of opportunity cost because you're being conservative but you're safeguard from the tail risk.
- Model using normal distribution will be accurate most of the time but you're exposed to tail risk which can be catastrophic.

In conclusion, knowing when and what distribution to use is what differentiate sophisticated trader from the average trader.
