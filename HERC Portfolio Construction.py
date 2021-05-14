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
# # HERC Portfolio Construction

# %%
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from cardano.data_query.data_client import DataClient
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import pdist
from finance_ml.clustering import get_quasi_diagonal_matrices, get_hrp_weights

plt.style.use("cardano")

# %%
# get data 

query = """
select 
    as_of_date,
    index_name,
    value
from hdpquantsandbox.sca_prices_index
"""

data_client = DataClient()
data_client.append("sca_data", query, "hive")
raw_data = data_client.request()["sca_data"]

# %%
# format data a bit

data = raw_data.copy(deep=True)
data["as_of_date"] = pd.to_datetime(data["as_of_date"])
data = pd.pivot_table(data, index="as_of_date", columns="index_name", values="value")
data = data.loc["2007-01-01":, :]
data = data.drop(["World Equity", "LIBOR", "US Equity Puts"], axis=1)

# %%
# convert data to 5 day rolling returns and get cov matrix

returns_data = (data / data.shift(5) - 1).dropna()
corr = returns_data.corr()

# %%
# plot the correlation matrix

fig, ax = plt.subplots()
sns.heatmap(corr, ax=ax, xticklabels=True, yticklabels=True)
ax.set_title("Standard correlation matrix", fontsize=20)
fig.set_size_inches(15, 15);

# %%
# get the link matrix 

dist = ((1 - corr) / 2) ** 0.5
dist = pdist(dist, metric="euclidean")
link = sch.linkage(dist, "average")

# %%
# plot the dendogram

fig, ax = plt.subplots()
fig.set_size_inches(20, 5)
ax.set_title("Dendrogram for clustering", fontsize=15)
sch.dendrogram(link, ax=ax, labels=data.columns);

# %%
corr_diag, cov_diag = get_quasi_diagonal_matrices(returns_data)
cov_diag *= np.sqrt(252)
fig, axs = plt.subplots(2, 1)
sns.heatmap(corr_diag, ax=axs[0], xticklabels=True, yticklabels=True)
axs[0].set_title("Quasi-diagonal correlation matrix", fontsize=20)
sns.heatmap(cov_diag, ax=axs[1], xticklabels=True, yticklabels=True)
axs[1].set_title("Quasi-diagonal covariance matrix", fontsize=20)
fig.set_size_inches(15, 25)
fig.tight_layout()

# %%
weights = get_hrp_weights(returns_data)
weights * 100

# %%
