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
# # 4. Sample Weights

# %% [markdown]
# In order to train a machine learning algorithm, we need to sample from our available data (potentially with replacement, if we are bootstrapping). A key issue with financial data is that we are generally interested in the outcome over a time period, and hence across our dataset many of our observations will in fact have some overlap. This causes an issue, because our observations are hence very much not **independently identically distributed**, but this is often a condition for machine learning algorithms to learn effectively.
#
# This notebook sets out some ideas that can be used to generate samples for training on that help to mitigate this problem to some degree.

# %%
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple

# %% [markdown]
# ## Uniqueness of labels

# %% [markdown]
# For a given label, we define the **uniqueness** at time t to be:
#
# 1) 0 if t is not in the interval covered by the label; or
#
# 2) the reciprocal of the number of labels which include the point t if it is covered by the label
#
# This is a measure of how much the label overlaps with others at a given point t, with a lower value indicating a greater deal of overlap (as its the reciprocal of the count).  We then define the **average uniqueness** for a point to be the sum of the uniqueness at each point covered by the label, divided by the number of points covered by the label.  Note that again a smaller number points to a greater deal of overlap. Note also that the average uniqueness must by defition lie in the interval (0, 1].
#
# In the example below, we create a dummy set of intervals, and calculate the average uniqueness for each.

# %%
# create dummy intervals

interval_starts = pd.date_range(start=datetime(2020, 1, 1), end=datetime(2020, 6, 30))
intervals = pd.Series(index=interval_starts, dtype="object")
rng = np.random.default_rng(1234)

for i in range(len(intervals)):
    n = rng.integers(1, 20)
    intervals.iloc[i] = intervals.index[i] + pd.tseries.offsets.Day(n)


# %% [markdown]
# The below function calculates the number of current events per bar - note that here we are assuming a bijection between the labels above and bars - however, in a real dataset the labelled ranges may be a subset of the total number of bars available.

# %%
def get_overlap_count(bar_dates: pd.Series,
                      intervals: pd.Series,
                      date_range: Tuple[datetime, datetime]) -> pd.Series:
    
    # first we need to process intervals a bit
    intervals = intervals.copy(deep=True)
    intervals = intervals.fillna(bar_data.iloc[-1])  # we must close off any open intervals with the date of the last bar
    intervals = intervals[intervals > date_range[0]]  # only need events that end after the start of our range
    intervals = intervals[intervals.index < date_range[1]]  # and only need events that start before the end of the range
    
    # now we can actually get our counts
    iloc = bar_dates.searchsorted([intervals.index[0], ])
    


# %%
date_range = [datetime(2020, 2, 1), datetime(2020, 5, 30)]
bar_dates = pd.Series(intervals.index)

# first we need to process intervals a bit
intervals = intervals.copy(deep=True)
intervals = intervals.fillna(bar_data.iloc[-1])  # we must close off any open intervals with the date of the last bar
intervals = intervals[intervals > date_range[0]]  # only need events that end after the start of our range
intervals = intervals[intervals.index < date_range[1]]  # and only need events that start before the end of the range

# now we can actually get our counts
iloc = bar_dates.searchsorted([intervals.index[0], intervals.max()])
count = pd.Series(0, index=bar_data[iloc[0]: iloc[1] + 1])

# %%
count

# %%
