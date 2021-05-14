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
# # Heirarchical clustering

# %% [markdown]
# The most famous technique for portfolio optimization is Markowitz mean-variance optimization. This is a quadratic optimization problem. With only equality constraints it can be solved using Lagrange multipliers - otherwise, Markowitz devised a special algorithm (the *Critical Line Algorithm*) to find a solution. The main issue is that a solution requires inverting the covariance matrix, which is generally ill-conditioned. The resulting portfolios are hence unstable. Also, Markowitz optimization tends to lead to highly concentrated portfolios.
#
# Marcos suggests an alternative approach based on heirarchical clustering of the asset universe. I explore this in detail in this notebook.

# %% [markdown]
# ## Dummy data

# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from warnings import simplefilter
import seaborn as sns


# %%
rng = np.random.default_rng(seed=120)
n_observations = 500

# group 1
asset_1 = rng.normal(loc=0, scale=0.1, size=(n_observations, 1))
asset_2 = rng.normal(loc=0, scale=0.1, size=(n_observations, 1)) * np.sqrt(1 - 0.7 ** 2) + 0.7 * asset_1
asset_3 = rng.normal(loc=0, scale=0.1, size=(n_observations, 1)) * np.sqrt(1 - 0.85 ** 2) + 0.85 * asset_1

# group 2
asset_4 = rng.normal(loc=0, scale=0.1, size=(n_observations, 1))
asset_5 = rng.normal(loc=0, scale=0.1, size=(n_observations, 1)) * np.sqrt(1 - 0.6 ** 2) + 0.6 * asset_4
asset_6 = rng.normal(loc=0, scale=0.1, size=(n_observations, 1)) * np.sqrt(1 - 0.9 ** 2) + 0.9 * asset_4

# group 3
asset_7 = rng.normal(loc=0, scale=0.1, size=(n_observations, 1)) * np.sqrt(1 - 0.2 ** 2) - 0.2 * asset_1

data = np.concatenate([asset_1, asset_2, asset_3, asset_4, asset_5, asset_6, asset_7], axis=1)

# %% [markdown]
# ## Clustering

# %% [markdown]
# The first step is for us to group the assets into clusters.

# %%
correlations = np.corrcoef(data.T)
corr_distance = np.sqrt(0.5 * ( 1 - correlations))
dist = pdist(corr_distance, metric="euclidean")
link = sch.linkage(dist, "single")
link

# %%
fig, ax = plt.subplots()
fig.set_size_inches(15, 5)
ax.set_title("Dendrogram for clustering", fontsize=15)
sch.dendrogram(link, ax=ax);


# %%
def get_quasi_diagonal(link_matrix):
    link_matrix = link_matrix.copy().astype(int)
    sorted_index = pd.Series([link_matrix[-1, 0], link_matrix[-1, 1]])
    total_assets = link_matrix[-1, -1]  # get total assets from link matrix bottom right
    while sorted_index.max() >= total_assets:  # still some grouped assets
        sorted_index.index = range(0, sorted_index.shape[0] * 2, 2)
        df_0 = sorted_index[sorted_index >= total_assets]
        i = df_0.index
        j = df_0.values - total_assets
        sorted_index[i] = link_matrix[j, 0]
        df_0 = pd.Series(link_matrix[j, 1], index=i+1)
        sorted_index = sorted_index.append(df_0)
        sorted_index = sorted_index.sort_index()
    
    sorted_index.index = range(sorted_index.shape[0]) 
    return sorted_index.to_list()

sorted_assets = get_quasi_diagonal(link)

# %%
sns.heatmap(pd.DataFrame(data[:, sorted_assets]).corr())


# %%
def get_group_var(cov,
                  sub_group):
    
    group_cov = cov[np.ix_(sub_group, sub_group)]
    inv_diag = 1 / np.diagonal(group_cov)
    ivp_weights = (inv_diag / inv_diag.sum())
    ivp_weights = ivp_weights.reshape(-1, 1)
    variance = (ivp_weights.T @ group_cov @ ivp_weights).squeeze()
    return variance


def get_recursive_bisection(cov,
                            sorted_assets):
    
    weights = pd.Series(1, index=sorted_assets)  # initialise all weights to 1
    all_sub_groups = [sorted_assets]
    while all_sub_groups:
        
        # bisect each subgroup
        bisected_sub_groups = []
        for sub_group in all_sub_groups:
            n = len(sub_group)
            if n > 1:
                bisected_sub_groups.extend([sub_group[:n // 2], sub_group[n // 2:]])
        
        # now update weights for each bisected group, handling them in pairs
        all_sub_groups = bisected_sub_groups
        for i in range(0, len(all_sub_groups), 2):
            sub_group_1 = all_sub_groups[i]
            sub_group_2 = all_sub_groups[i + 1]
            group_var_1 = get_group_var(cov, sub_group_1)
            group_var_2 = get_group_var(cov, sub_group_2)
            a = 1 - group_var_1 / (group_var_1 + group_var_2)
            weights[sub_group_1] *= a  # update weights for group 1
            weights[sub_group_2] *= (1 - a)  # update weights for group 2
        
    return weights
        
        

# %%
sorted_data = data[:, sorted_assets]
cov = np.cov(sorted_data, rowvar=False)
get_recursive_bisection(cov, sorted_assets)

# %%
