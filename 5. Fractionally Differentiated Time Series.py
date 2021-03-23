# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: machine-learning
#     language: python
#     name: machine-learning
# ---

# %% [markdown]
# # Fractionally differentiated time series

# %% [markdown]
# Generally, time series modelling in finance follows one of two paradigms: **Box-Jenkins** wherein we difference a series an integer amount of times to get a stationary series, or **Engle-Granger**, where we work with the prices of cointegrated series (ie non-stationary series which are stationary as a linear combination).  The latter removes almost all memory from the series and so we lose information, while the requirement to use cointegrated series in the latter is highly restrictive.
#
# The proposal is to *fractionally difference* the series, where we use the real valued exponent version of the binomial expansion to write a differenced series as the weighted sum or prior values.  Clearly we can't have an infinite lookback so we fix a window such that sufficiently small weights are ignored.

# %%
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot as plt

# %matplotlib inline

# %% [markdown]
# ## Example 1 - random walk

# %%
rng = np.random.default_rng()
rnd = rng.normal(size=1000)
pd.Series(rnd).plot(kind="line");
plt.gcf().set_size_inches(15, 5)
plt.gca().set_title("Gaussian random variables", fontsize=20);

# %%
adf, p, *_ = adfuller(rnd)
print(f"Dickey fuller stat of time series is {round(adf, 2)} vs 5% critical value of -2.86")

# %% [markdown]
# As we would expect, the augmented Dickey-Fuller test is telling us that this is stationary. Next we do the same with the cumulative sum - this is a random walk which is **NOT** a stationary process as variance increases over time.

# %%
cum_rnd = rnd.cumsum()
pd.Series(cum_rnd).plot(kind="line");
plt.gcf().set_size_inches(15, 5)
plt.gca().set_title("Gaussian random walk", fontsize=20);

# %%
adf, p, *_ = adfuller(cum_rnd)
print(f"Dickey fuller stat of time series is {round(adf, 2)} vs 5% critical value of -2.86")
print(f"The p value of the test is {'{:.3%}'.format(p)}")

# %% [markdown]
# The augmented Dickey-Fuller test is suggesting we have a unit root (ie non-stationarity of the time series), again as we might expect. This series should have an order of integration of 1 - suppose instead that we *over-difference* the series by differencing it twice.

# %%
rnd_diff = np.diff(np.diff(cum_rnd))  # difference twice
pd.Series(rnd_diff).plot(kind="line");
plt.gcf().set_size_inches(15, 5)
plt.gca().set_title("Over-differenced series", fontsize=20);

# %%
adf, p, *_ = adfuller(rnd_diff)
print(f"Dickey fuller stat of time series is {round(adf, 2)} vs 5% critical value of -2.86")

# %% [markdown]
# We note that the Dickey Fuller test statistic is (while still well within the critical range to reject the null hypothesis) lower than for our initial series. We have made things worse by over-differencing the series.

# %% [markdown]
# ## Example 2 - sinusoidal function

# %% [markdown]
# Here, we try a series which has memory, rather than the memoryless random walk series we considered above.

# %%
sinusoid = np.sin(np.linspace(0, 20, 1000))
pd.Series(sinusoid).plot(kind="line");
plt.gcf().set_size_inches(15, 5)
plt.gca().set_title("Sinusoidal function", fontsize=20);

# %%
adf, p, *_ = adfuller(sinusoid)
print(f"Dickey fuller stat of time series is {round(adf, 2)} vs 5% critical value of -2.86")

# %% [markdown]
# As we would expec, the test suggests (very strongly) that the sine wave is stationary. However, we note that this does not imply that the series has no memory. This is is some ways the converse of our previous example, where we had a non-stationary series with no memory.
#
# Suppose that we now add a trend - what will this do?

# %%
trend = np.linspace(0, 20, 1000) * 0.01
sinusoid_trend = (sinusoid + trend).cumsum()
pd.Series(sinusoid_trend).plot(kind="line");
plt.gcf().set_size_inches(15, 5)
plt.gca().set_title("Sinusoidal function with trend", fontsize=20);

# %%
adf, p, *_ = adfuller(sinusoid_trend)
print(f"Dickey fuller stat of time series is {round(adf, 2)} vs 5% critical value of -2.86")
print(f"The p value of the test is {'{:.3%}'.format(p)}")


# %% [markdown]
# As expected, our augmented Dickey-Fuller test now suggests very strongly that there is a trend. Lets try applying some fractional differencing to remove this trend.

# %%
def get_weights(d: float,
                size: int) -> np.array:
    """
    get weights for fractional binomial
    expansion
    """
    
    w = [1.]
    for k in range(1, size):
        w_next = -1 * w[-1] * (d - k + 1) / k
        w.append(w_next)
    
    return np.array(w[::-1]).reshape(-1, 1)


def expanding_window_fd(time_series: pd.DataFrame,
                        d: float,
                        threshold: float = 0.01) -> np.array:
    """
    carry out fractional differencing of order d
    on the given time series to remove/reduce trend
    """
    
    all_weights = get_weights(d, time_series.shape[0])
    cumulative_weights = all_weights.cumsum()
    weight_capture =  cumulative_weights / cumulative_weights[-1]
    skip = (weight_capture > threshold).sum()
    
    out = dict()
    for name in time_series.columns:
        filled_series = time_series[name].fillna(method="ffill").dropna()
        diffed_series = pd.Series(dtype=float)
        for i in range(skip, filled_series.shape[0]):
            loc = filled_series.index[i]
            if not np.isfinite(time_series.loc[loc, name]):
                continue  # don't compute weight if NaN in original time series
            weights = all_weights[-(i + 1):, :]
            diffed_series.loc[loc] = (weights.T @ filled_series.loc[:loc]).squeeze()
        
        out[name] = diffed_series.astype("float")
    
    return pd.DataFrame(out)    


# %% [markdown]
# The below suggests that we require d=1 to remove the linear trend

# %%
time_series = pd.Series(sinusoid_trend).to_frame()
fig, ax = plt.subplots()

for d in [1.01, 0.99, 0.9, 0.5, 0.35, 0.2, 0.1, 0.05]:
    diffed_series = expanding_window_fd(time_series.copy(), d)
    adf, p, *_ = adfuller(diffed_series)
    diffed_series.columns = [f"Diffed with {d} ({p}, {adf})"]
    diffed_series.plot(kind="line", ax=ax)

pd.Series(sinusoid_trend).plot(kind="line", label="original", linestyle="--");
fig.set_size_inches(15, 5)

# %% [markdown]
# ## Example 3 - S&P daily prices

# %% [markdown]
# For a more realistic example, we can take a look at S&P daily returns. Here, we will apply a fixed-width window as suggested by M. Lopez de Prado rather than the expanding window that we have used above.

# %%
s_and_p = pd.read_csv("spx.csv", index_col=0)
s_and_p.columns = ["S&P 500"]
s_and_p = s_and_p.fillna(method="ffill")
s_and_p.plot(kind="line")
plt.gcf().set_size_inches(15, 5)

# %%
adf, p, *_ = adfuller(s_and_p)
print(f"Dickey fuller stat of time series is {round(adf, 2)} vs 5% critical value of -2.86")
print(f"The p value of the test is {'{:.3%}'.format(p)}")


# %% [markdown]
# As we can see, the series does not at all look stationary. This is confirmed by the augmented Dickey-Fuller technique.

# %%
def get_weights_fixed(d: float, 
                      threshold: float) -> np.array:
    """
    get weights for fractional differencing for a
    fixed size window
    """
    
    w = [1]
    k = 1
    while True:
        next_w = -1 * w[-1] * (d - k + 1) / k
        if abs(next_w) < threshold:
            break
        else:
            w.append(next_w)
            k += 1 
    
    return np.array(w).reshape(-1, 1)


def fixed_window_fd(time_series: pd.DataFrame,
                    d: float,
                    threshold: float = 1e-5) -> np.array:
    """
    carry out fractional differencing of order d
    on the given time series to remove/reduce trend
    """
    
    weights = get_weights_fixed(d, threshold)  
    interval = len(weights) - 1
    out = dict()
    
    for name in time_series.columns:
        filled_series = time_series[name].fillna(method="ffill").dropna()
        diffed_series = pd.Series(dtype=float)
        for i in range(interval, filled_series.shape[0]):
            start, end = filled_series.index[i - interval], filled_series.index[i]
            if not np.isfinite(time_series.loc[end, name]):
                continue  # don't compute weight if NaN in original time series
            diffed_series.loc[end] = (weights.T @ filled_series.loc[start:end]).squeeze()
        
        out[name] = diffed_series.astype("float")
    
    return pd.DataFrame(out) 



# %%
fig, ax = plt.subplots()

for d in [0.5, 0.35, 0.2]:
    diffed_series = fixed_window_fd(s_and_p.copy(), d)
    adf, p, *_ = adfuller(diffed_series)
    diffed_series.columns = [f"Diffed with {d} ({p}, {adf})"]
    diffed_series.plot(kind="line", ax=ax)

s_and_p.plot(kind="line", label="S&P 500 (original)", linestyle="--", ax=ax);
fig.set_size_inches(15, 5)

# %% [markdown]
# Here we can see that a value of d=0.35 works pretty well for differencing the series. This gets us to a series which looks pretty stationary, but at the same time has not lost all of the memory embedded in the original series.

# %%
