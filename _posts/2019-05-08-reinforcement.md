---
title: "Reinforcement Learning"
date: 2019-05-08
tags: [machine learning, reinforcement learning]
excerpt: "Basic reinforcement learning"
---

```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
# Q3a ----------------------
T = 1000

def e_greedy(Q, N, t, e):
    p = np.random.uniform()
    
    if p > e:
        A = np.argmax(Q)
        return A
    else:
        A = np.random.randint(0,10)
        return A

    
def UCB(Q, N, t, c):
    new_Q = Q + c*np.sqrt(np.log(t)/N)
    A = np.argmax(new_Q)
    
    return A
```


```python
# Q3b ----------------------
def test_run(policy, param):
    true_means = np.random.normal(0,1,10)
    reward = np.zeros(T+1)
    Q = np.zeros(10)
    N = np.ones(10)
    
    for t in range(1,T+1):
        a = policy(Q,N,t,param)
        
        r = np.random.normal(true_means[a],1)
        reward[t] = r
        N[a] = N[a] + 1
        Q[a] = Q[a] + (r-Q[a])/N[a]
        
    return reward
```


```python
# Q3c -----------------------
# e_greedy switch from better to worse than greedy when
# epsilon > 0.3
def main():
    ave_g = np.zeros(T+1)
    ave_eg = np.zeros(T+1)
    ave_ucb = np.zeros(T+1)
    
    for i in range(2000):
        g = test_run(e_greedy, 0.0)
        eg = test_run(e_greedy, 0.2)
        ucb = test_run(UCB, 2)
        
        ave_g += (g-ave_g)/(i+1)
        ave_eg += (eg-ave_eg)/(i+1)
        ave_ucb += (ucb-ave_ucb)/(i+1)
        
    t = np.arange(T+1)
    plt.plot(t, ave_g, 'b-', t, ave_eg, 'r-', t, ave_ucb, 'g-')
    plt.show()
```


```python
main()
```


<img src="/images/2019-05-08-reinforcement/output_4_0.png",alt="policies">

