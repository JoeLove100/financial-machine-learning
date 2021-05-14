# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # Meta-labelling

# %% [markdown]
# This notebook deals with exercise 3.4 from the book "*Advances in Financial Machine Learning*".  We have an algorithm to suggest when we would like to go long/short (here we use a simple trend indicator) and we wish to train an algorithm to learn whether we should take the trade or not. 

# %% [markdown]
# ## Step 1 -  Read in our data

# %% [markdown]
# Here we are choosing to use daily price returns on the S&P 500, and a basic momentum strategy that goes long when the 50D MA is above the 200D MA (and shorter when it is below).  This gives the direction of the trade, and we then want to train an algorithm to tell us whether we should trade or not.  The first step is to read in our data and to create some required columns. We also make a plot of the diff of our moving averages vs the actual level of the index to illustrate the relationshipt we are dealing with.

# %%
import pandas as pd
import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

np.random.seed(2020)

# %%
close_data = pd.read_csv("raw_data/spx.csv", index_col=0)
close_data.index = pd.to_datetime(close_data.index)
close_data["PX_LAST"] = close_data["PX_LAST"].ffill()
close_data["50DMA"] = close_data["PX_LAST"].rolling(50).mean()
close_data["200DMA"] = close_data["PX_LAST"].rolling(200).mean()

# %%
fig, ax1 = plt.subplots()
ax2 = ax1.twinx() 
close_data["PX_LAST"].plot(kind="line", ax=ax1)
(close_data["50DMA"] - close_data["200DMA"]).plot(kind="line", ax=ax2, color="red")
fig.legend()
fig.set_size_inches(15, 5)

# %%
close_data = close_data.dropna(how="any", axis=0)


# %% [markdown]
# ## Step 2 - Put data into "bins"

# %% [markdown]
# First, we need to market each bar as a winner or a loser - here we have two options:
#
# 1) we can first use an ML algorithm to work out which side we should have taken in the asset (ie gone long or short), and then run a second pass through to work out whether we would have made or lost money based on these trades.  This takes a purely ML approach, with ML telling us both how we trade and when we should trade.
#
# 2) somewhat simpler, we can use another trading rule (eg our basic momentum strategy outlined in the previous section) which tells us which side of the bet we should take. Then, we can use a pass through the functions in this section in order to mark each section as a "win" or a "loss".  We can train an ML algorithm on this to tell us when we should or should not trade.
#
# The beauty of this approach is that we can combine a simple non-ML signal for going long/short with a more complex ML modes which can evaluate whether we should actually listen to the signal or not. THe term "meta-labelling" comes from the fact that these secondary labels depend on the output of this first model.
#
# The approach here is more sophisticated than just "was the return +ve or -ve over the next n periods?".  We have three barriers that can be hit - a stop-loss barrier, a profit-take barrier, and a max time period barrier. We then take the first of these that is hit and work out what the return to that point was, and mark 0 (don't invest, it loses money) or 1 (do invest, it makes money) based on the outcome.

# %%
def apply_triple_barrier(close: pd.Series,
                         events: pd.DataFrame,
                         profit_take_stop_loss: Tuple[float, float]) -> pd.DataFrame:
    """
    apply the 'triple barrier' method to our 
    """
    
    # set up an output vector
    out = events["max_time"].copy(deep=True).to_frame()
    out["stop_loss"] = None
    out["profit_take"] = None

    # 1) set up our series of profit take limits 
    if profit_take_stop_loss[0] > 0:
        profit_take = events["target"] * profit_take_stop_loss[0]
    else:
        profit_take = pd.Series(index=events.index)
    
    # 2) similarly, set up the stop loss 
    if profit_take_stop_loss[1] > 0:
        stop_loss = -events["target"] * profit_take_stop_loss[1]
    else:
        stop_loss = pd.Series(index=events.index)
    
    # 3) iterate over our events and label each one
    events["max_time"] = events["max_time"].fillna(close.index[-1])  # fill missing with last close price date
    for start, end in events["max_time"].iteritems():
        prices = close.loc[start: end]
        returns = (prices / prices[0] - 1) * events.at[start, "side"]
        out.loc[start, "stop_loss"] = returns[returns <= stop_loss[start]].index.min()
        out.loc[start, "profit_take"] = returns[returns >= profit_take[start]].index.min()

    out["stop_loss"] = pd.to_datetime(out["stop_loss"])
    out["profit_take"] = pd.to_datetime(out["profit_take"])
    return out


# %%
def get_labelled_events(close: pd.Series,
                        selected_dates: pd.Series,
                        profit_take_stop_loss: Tuple[float, float],
                        target: pd.Series,
                        min_return: float,
                        finish_time: pd.Series = None,
                        side: pd.Series = None) -> pd.DataFrame:
    """
    label each "event" based on whether we make or lose 
    money if we have provided side or just which boundary
    which we hit first if not
    """
    
    # 1) get targets we are actually interested in - these will be the examples we want to trai n on
    target = target.loc[selected_dates]
    target = target[target > min_return]
    
    # 2) if no max holding periods provided, just set these to be the last closing price date
    if finish_time is None:
        finish_time = pd.Series(max(close.index), index=target.index)
    
    # 3) now set the arbitrarily initialise side to all long if not defined
    if side is None:
        side = pd.Series(1, index=target.index)
        profit_take_stop_loss = [profit_take_stop_loss[0], profit_take_stop_loss[1]]  # symmetric if learning side
    
    # 4) now create our events object
    events = pd.concat([finish_time, target, side], axis=1, keys=["max_time", "target", "side"])
    events = events.dropna(subset=["target"])
    labelled_events = apply_triple_barrier(close, events, profit_take_stop_loss)
    # labelled_events = labelled_events.drop(["side"], axis=1)
    labelled_events["first"] = labelled_events.min(axis=1)
    return labelled_events


# %%
def get_bins(labelled_events: pd.DataFrame,
             close: pd.Series) -> pd.DataFrame:
    
    all_dates = labelled_events.index.union(labelled_events["first"]).drop_duplicates()
    filled_close = close.reindex(all_dates, method="bfill")
    out = pd.DataFrame(index=labelled_events.index)
    out["return"] = filled_close.loc[labelled_events["first"]].values / filled_close.loc[labelled_events.index] - 1
    if "side" in labelled_events:
        out["return"] *= out["side"]
        out["bin"] = np.where(out["return"] > 0, 1, 0)
    else:
        out["bin"] = np.where(out["return"] > 0, 1, -1)

    return out


# %% [markdown]
# Now we have defined our functions, we set up our actual series so that we can create our bins.  Note that we can specify the profit take and stop loss limits separately, and also that the "target" that these are based on can vary over time (so we can have it adapt as vol changes over time).  We then use these to mark our bars using the get_bins function.

# %%
close = close_data.loc[:, "PX_LAST"]  # define closing price data
selected_dates = pd.Series(np.random.choice(close_data.index, int(0.6*len(close)), replace=False))  # 60% of data as training
profit_take_stop_loss = (1, 2)  # profit take is 1 * target up, while stop loss is 2 * target down
target = (close / close.shift(1) - 1).rolling(10).std().bfill() * np.sqrt(10)  # let target be rolling 10 day vol
min_return = 0.01  # ignore any entries where the target is too low to be material
finish_time = pd.Series(close.index + np.timedelta64(3, "M"), index=close.index)  # max 3 month holding period
side = np.sign(close_data["50DMA"] - close_data["200DMA"])  # go long if 50D MA > 200D MA, otherwise short
events = pd.concat([finish_time, target, side], axis=1, keys=["max_time", "target", "side"])  # combine some of the above

# %%
labelled_events = get_labelled_events(close, selected_dates, profit_take_stop_loss, target,
                                      min_return, finish_time, side)
bins = get_bins(labelled_events, close)
# %% [markdown]
# ## Step 3 - Train an ML algorithm

# %% [markdown]
# We have now marked our training sets with 0 and 1, so we know whether we would have made or lost money by following the strategy at each point. Our next job is to train an ML algorithm to tell us when we should trade and when we should not. Here we are going to construct a set of common indicators based on the close price to train our algorithm on

# %%
# set up our features

data = close.copy(deep=True).to_frame().sort_index()
data["1_month"] = close / close.shift(21)
data["3_month"] = close / close.shift(42)
data["6_month"] = close / close.shift(126)
data["1_month_std"] = data["1_month"] / (close.rolling(21).std() * np.sqrt(21))
data["3_month_std"] = data["3_month"] / (close.rolling(42).std() * np.sqrt(42))
data["6_month_std"] = data["6_month"] / (close.rolling(126).std() * np.sqrt(126))
data["macd_short"] = close.ewm(span=8).mean()  - close.ewm(span=24).mean()
data["macd_short"] = close.ewm(span=16).mean()  - close.ewm(span=48).mean()
data["macd_short"] = close.ewm(span=32).mean()  - close.ewm(span=96).mean()
data = data.drop("PX_LAST", axis=1)

# %%
# split into test and training

training_data = data.loc[selected_dates].sort_index()
training_data = training_data.dropna(how="any")
training_data = pd.merge(training_data, bins["bin"], left_index=True, right_index=True)

test_data = data[~data.index.isin(selected_dates)].sort_index()
test_data = test_data.dropna(how="any")

# %%
# check shape

training_data.shape, test_data.shape

# %%
# now train our classifier

classifier = RandomForestClassifier()
classifier.fit(training_data.drop("bin", axis=1), training_data["bin"])
test_data["predictions"] = classifier.predict(test_data)

# %%
classifier.score(training_data.drop("bin", axis=1), training_data["bin"])

# %%
