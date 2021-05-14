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
# # Portfolio Optimisation

# %% [markdown]
# This notebook provides an overview of several popular methods for portfolio construction. To illustrate these, I have used the market data series which make up Cardano's own (pre-2021 review) SCA, which provides a decent proxy for the macro investment toolkit that we have available to us.

# %% [markdown]
# ## Setting up 

# %% [markdown]
# We need to import some common libraries in order to produce our analysis - I have also stored some of the more complex functions that we require in a separate .py file to clean up the notebook a bit, so these are imported also.  
#
# We also download the required SCA returns data from Hive, using daily price data from 1 Jan 2004 to 31 March 2021. Note that we use 5-day rolling returns, as we have found that these provide better estimates of the covariance between different assets than daily returns, which are unduly influenced by timezone issues.  For the expected returns, we use assumptions from our central economic scenario as of 31 March 2021.

# %%
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, ticker as mtick
from cardano.data_query.data_client import DataClient
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import pdist
from finance_ml.clustering import get_quasi_diagonal_matrices, get_hrp_weights, get_global_minimum_weights, \
   get_equal_risk_contribution_weights, get_gap_metric, get_herc_weights

plt.style.use("cardano")

# %%
# get SCA returns and weights data  

returns_query = """
select 
    as_of_date,
    index_name,
    value
from hdpquantsandbox.sca_prices_index
"""

weights_query = """
select 
    index_name,
    weight
from hdpquantsandbox.sca_weight_index
where as_of_date = "2021-03-31"
and strategy_name = "SCA"
and version = 4
"""

data_client = DataClient()
data_client.append("sca_returns", returns_query, "hive")
data_client.append("sca_weights", weights_query, "hive")
sca_data = data_client.request()
raw_data = sca_data["sca_returns"]
sca_weights = sca_data["sca_weights"]

# %%
# format SCA data a bit

data = raw_data.copy(deep=True)
data["as_of_date"] = pd.to_datetime(data["as_of_date"])
data = pd.pivot_table(data, index="as_of_date", columns="index_name", values="value")
data = data.loc["2004-01-01":, :]
data = data.drop(["World Equity", "LIBOR", "US Equity Puts", 
                  "Global High Div Equity", "Global High Yield"], axis=1)

sca_weights = sca_weights.set_index("index_name")
sca_weights = sca_weights.reindex(data.columns)

# %%
# read in expected return data

expected_returns = pd.read_csv("raw_data/expected_returns.csv")
expected_returns = expected_returns.set_index("Asset")
expected_returns = expected_returns.reindex(data.columns)

# %%
# convert data to 5 day rolling returns and get cov matrix

returns_data = (data / data.shift(5) - 1).dropna()
corr = returns_data.corr()

# %%
# plot the correlation matrix

fig, ax = plt.subplots()
sns.heatmap(corr, ax=ax, xticklabels=True, yticklabels=True)
ax.set_title("Correlation matrix - 5d returns (2004 - present)", fontsize=20)
ax.set_xlabel("")  # no label at bottom
ax.set_ylabel("")  # no label on LHS
fig.set_size_inches(15, 15);

# %% [markdown]
# ## Markowitz portfolio optimisation

# %% [markdown]
# Standard mean-variance (Markowitz) portfolio optimisation requires us to find the portfolio with the lowest level of risk for a given desired return, where the risk is defined to be the variance of the portfolio. Clearly, this requires us to provide both a vector of expected returns for our assets, and also a complete covariance matrix.
#
# The problem is generally formulated as a quadratic optimisation with a number of equality and innequality constraints - Markowitz provided an algorithm to solve this general case (Critical Line Algorithm), or it can be solved by general purpose QP software. However, I here only consider the simple case in which we wish to find the portfolio with the maximum Sharpe ratio subject to the portfolio being fully invested.
#
# If we have expected returns $\mu$ and covariance matrix $\Sigma$, the our max Sharpe weights $\hat{w}$ are given by the formula
#
# $\hat{w} = \frac{\Sigma^{-1}\mu}{e^{T}\Sigma^{-1}\mu}$
#
# where $e$ denotes a vector of ones.
#

# %%
mu = expected_returns["Expected return"].values
e = np.ones(shape=(corr.shape[0], 1))
cov = returns_data.cov().values
cov_inv = np.linalg.inv(cov)

numerator = cov_inv @ mu
denominator = e.T @ cov_inv @ mu
weights = pd.Series(numerator / denominator, index=data.columns)

# %%
fig, ax = plt.subplots()
weights.plot(kind="bar")
ax.yaxis.set_major_formatter(mtick.FuncFormatter("{:.0%}".format))
ax.set_title("Markowitz portfolio weights", fontsize=20)
fig.set_size_inches(15, 5)

# %% [markdown]
# The portfolio suggested is wildly impractical and highly concentrated. Even in the more sophisticated case wherein we specify maximum and minimum holdings in different assets, Markowitz portfolio optimisation has a number of recognised limitations:
#
# * It tends to produce highly concentrated portfolios. Intuitively, given the uncertainty around the values of the expected returns, it seems undesirable for such a method to produce portfoilos which represent a significant "bet" on a particular state or states of the world.
# * When we require diversification most is when we have a number of highly correlated assets. However, in this situation our covariance matrix will be highly ill-conditioned, and so (because the formula for our optimal portfolio requies that we invert the covariance matrix) small changes in the inputs can lead to huge changes in the suggested weights.
# * coming up with accurate expected returns is of course incredibly challenging, and the inherent uncertainty in this is at odds with the highly concentracted portfolios that this method tends to produce.

# %% [markdown]
# ## Black-Litterman portfolio optimisation 

# %% [markdown]
# As an attempt to improve on the above, the Black-Litterman method was developed by Fischer Black and Robert Litterman. The starting point for this method is to hold the "market portfolio", in which each asset is held in proportion to its size as a % of the global investable universe. The method the provides a formula for including a portfolio managers own views as an overlay to the allocations.
#
# I have not replicated this method here, as the data and calculations required are quite involved. However, an implementation is available on my personal GitHub [here](https://github.com/chief-gucci-sosa/black-litterman). A good overview of the method, and novel suggestion for setting the confidence of discretionary views in the model, can be found in "*A Step-by-Step Guide to the Black-Litterman Model*" by Thomas Idzorek. For a more mathematical treatment, see "*Mathematical Derivations and Practical Implications for the use of the Black-Litterman Model*" by Charlotta Mankert and Michael J Seiler. 
#
# While this method tends to produce more stable and less concentrated portfolios than pure Markowitz portfolio optimisation, and avoids issues of ill-conditioned covariance matrices, it still suffers some drawbacks. The relative complexity of the model and lack of academic evidence for out-of-sample outperformance may have hindered the uptake. Further, estimating the weights for the "market portfolio" on an ongoing basis is a difficult task.

# %% [markdown]
# ## Minimum variance portfolio

# %% [markdown]
# To avoid the difficult task of forecasting asset returns, we can instead search for the global minimum variance portfolio. On the sole condition that weights sum to 100%, this has the closed form solution as follows:
#
# $w = \frac{\Sigma^{-1}e}{e^{T}\Sigma^{-1}e}$
#
# However, I here wish to apply the additional constraint that all individual weights lie between 0 and 1, and hence will use a numerical optimization routine from the scipy package to find a solution.

# %%
weights = get_global_minimum_weights(returns_data.cov())

# %%
fig, ax = plt.subplots()
weights.plot(kind="bar")
ax.yaxis.set_major_formatter(mtick.FuncFormatter("{:.0%}".format))
ax.set_title("Global min variance portfolio weights", fontsize=20)
fig.set_size_inches(15, 5)

# %% [markdown]
# ## 1/n portfolio construction

# %% [markdown]
# A more simplistic attempt to circumvent the difficulties encountered in the MPT approaches above is the 1/n portfolio, where we just invest equal capital in each asset. While this provides a simple rule for building a diversified portfolio, it failes to account for either the different volatitlies associated with individual assets or the way that these assets behave in combination.
#
# Weights are shown below for completeness.

# %%
weights = pd.Series(np.ones(len(returns_data.columns)) / len(returns_data.columns), index=returns_data.columns)

# %%
fig, ax = plt.subplots()
weights.plot(kind="bar")
ax.yaxis.set_major_formatter(mtick.FuncFormatter("{:.0%}".format))
ax.set_title("1/n portfolio weights", fontsize=20)
fig.set_size_inches(15, 5)

# %% [markdown]
# ## Inverse volatility portfolio construction

# %% [markdown]
# Given the difficulty in estimating expected returns, many portfolio construction methods seek instead to build diversified portfolios using only the covariance matrix. A common idea is some notion of risk being shared equally across assets in the portfolio - the theory goes that this should provide an optimally diversified portfolio. Generally, such a portfolio will be leveraged in order for it to hit an investors required return, although the weights I have shown in this notebook assume 100% investment only.
#
# One of the simplest such strategies is the inverse volatility method - here, each asset is held so as to contribute equally to the gross volatility of the portfolio.

# %%
inverse_vols = 1 / np.sqrt(np.diag(returns_data.cov()))
total_inverse_vols = np.sum(inverse_vols)
weights = pd.Series(inverse_vols / total_inverse_vols, index=data.columns)

# %%
fig, ax = plt.subplots()
weights.plot(kind="bar")
ax.yaxis.set_major_formatter(mtick.FuncFormatter("{:.0%}".format))
ax.set_title("Inverse volatility portfolio weights", fontsize=20)
fig.set_size_inches(15, 5)

# %% [markdown]
# ## Equal risk contribution portfolio construction

# %% [markdown]
# A somewhat more sophisticated version of the above is given by "equal risk contribution" portfolio construction. We note that the volatility for weights $w$, denoted $\sigma(w)$, is a homogenous function of order 1 and hence by Euler's theorem for homogenous functions we have
#
# $\sigma(w) = \sum^{N}_{i=1} w_{i}\frac{\partial{\sigma(w)}}{\partial{w_{i}}}$
#
# Defining these terms as the risk contributions for each asset, to get an equaly contribution for each asset we then solve 
#
# $\min \sum^{N}_{i=1} [w_{i} - \frac{\sigma(w)^{2}}{(\Sigma w)_{i} N}]^{2}$
#
# subject to the condition that the weights sum to 1.
#
# It can be shown that this is closely related to finding the minimum variance portfolio - in fact, finding the minimum variance portfolio is equivalent to making the *marginal* rather than total risk contributions from each asset equal (see [this](https://www.grahamcapital.com/Equal%20Risk%20Contribution%20April%202019.pdf) note from US hedge fund Graham Capital to see why this is the case).  Intuitively, it can be though of as sitting somewhere inbetween the 1/n portfolio and the global minimum variance portfolio.

# %%
weights = get_equal_risk_contribution_weights(returns_data.cov())

# %%
fig, ax = plt.subplots()
weights.plot(kind="bar")
ax.yaxis.set_major_formatter(mtick.FuncFormatter("{:.0%}".format))
ax.set_title("Equal risk contribution portfolio weights", fontsize=20)
fig.set_size_inches(15, 5)

# %% [markdown]
# ## Economically balanced portfolio construction

# %% [markdown]
# A criticism of the above risk parity approaches is that they are too reliant on providing statistical diversification - rather than diversified exposure to the underlying drivers of markets, these portfolios seek to provide diversification based on realised asset returns (and the relationships between them) over some period. As these relationships can change over time, a more heuristic approach targeting risk allocations across key market drivers (in particular: growth, inflation, discount rates and risk premia) may be more appropriate. 
#
# This is the appraoch taken in Cardano's "Strategic Cluster Allocation", the weights of which are shown below - these weights predate the move to fixed notionals, but the philosophy is the same regardless of iplementation details. For a more quantitative take on this approach, see the paper "*Diversifying Macroeconomic Factors for Better of for Worse*" by Livia Amato and Harald Lohre of Invesco for some interesting ideas.
#
# Weights as of 31 March 2021 are shown for the SCA - note that I have rescaled these to 100% to be comparable with the other strategies in this document, but in reality the portfolio is run on a leveraged basis.

# %%
weights = sca_weights / sca_weights.sum()

# %%
fig, ax = plt.subplots()
weights.plot(kind="bar", ax=ax)
ax.yaxis.set_major_formatter(mtick.FuncFormatter("{:.0%}".format))
ax.set_title("Economicall balanced portfolio weights", fontsize=20)
fig.set_size_inches(15, 5)

# %% [markdown]
# ## Hierarchical Risk Parity

# %% [markdown]
# Heirarchical risk parity is an alternative portfolio construction technique proposed by Marcos Lopez de Prado in the paper "*Building Diversified Portfolios that Outperform Out-of-Sample*".  This portfolio construction technique works in three steps:
#
# 1) Use an agglomerative clustering algorith to sequentially group assets together based on the realised correlation matrix
#
# 2) Based on the groupings, reorder the assets so that the similar assets are next to each other - this has the effect of making your correlation and covariance matrices look almost block-diagonal.
#
# 3) Carry out a recursive bisection algorithm on the sorted lists, adjusting the weights each time in proportion to their inverse volatility weight risk contributions, so as to downweight riskier groups and upweight less risky groups
#
# The claimed advantage of this approach is that it gets around the problem of ill-conditioned covariance matrices entirely, as through our clustering we only consider related assets to be potential substitutes for each other (rather than all assets).

# %% [markdown]
# #### Step 1 - Cluster the assets

# %% [markdown]
# The first step is to get a "link matrix" describing how the assets should be clustered. In order to do so, we convert our correlation matrix to "correlation distances" through the formula:
#
# $d_{ij} = \sqrt{\frac{1 - \sigma_{ij}}{2}}$
#
# We then take the distance between two assets for the purposes of clustering to be the euclidean difference between the two assets' vectors of correlation distances.  We use single linkage for clustering purposes (ie when two assets are clustered, we take the min of the correlatin distances on combining them) as suggested in "*Advances in Financial Machine Learning*".

# %%
# get the link matrix 

dist = ((1 - corr) / 2) ** 0.5
dist = pdist(dist, metric="euclidean")
link = sch.linkage(dist, "single")

# %%
# plot the dendogram

fig, ax = plt.subplots()
fig.set_size_inches(20, 5)
ax.set_title("Dendrogram for clustering", fontsize=15)
sch.dendrogram(link, ax=ax, labels=data.columns);

# %% [markdown]
# #### Step 2 - Reorder the assets

# %% [markdown]
# The next step is to sequentially unpack the clustered assets, so that we end up with a quasi-diagonal matrix. The correlation and (annualised) covariance matrices are shown below after this procedure has been carried out.

# %%
corr_diag, cov_diag = get_quasi_diagonal_matrices(returns_data)
cov_diag *= np.sqrt(252 / 5)
fig, axs = plt.subplots(2, 1)
sns.heatmap(corr_diag, ax=axs[0], xticklabels=True, yticklabels=True)
axs[0].set_title("Quasi-diagonal correlation matrix", fontsize=20)
sns.heatmap(cov_diag, ax=axs[1], xticklabels=True, yticklabels=True)
axs[1].set_title("Quasi-diagonal covariance matrix", fontsize=20)
fig.set_size_inches(15, 25)
fig.tight_layout()

# %% [markdown]
# #### Step 3 - Calculate the weights

# %% [markdown]
# Finally, we wish to use the quasi-diagonalized covariance matrices to generate weights. In order to do so, we recursively bisect intervals, and reweight the two groups based on their risk. The risk is defined to be the variance of an inverse volatility portfolio in the assets.

# %%
weights = get_hrp_weights(returns_data)

# %%
fig, ax = plt.subplots()
weights.plot(kind="bar", ax=ax)
ax.yaxis.set_major_formatter(mtick.FuncFormatter("{:.0%}".format))
ax.set_title("HRP portfolio weights", fontsize=20)
fig.set_size_inches(15, 5)

# %% [markdown]
# ## Heirarchical Equal Risk Contribution

# %% [markdown]
# HERC is an innovation on HRP which makes three main changes to the approach:
#
# 1) When clustering the assest we use "ward" rather than "single" linkage
#
# 2) We use the gap statistic, described by Robert Tibshirani et al in the paper "*Estimating the Number of Clusters in a Data Set vai the Gap Statistic*", to work out an optimal number of clusters
#
# 3) Rather than a recursive bisection, we first weight between the clusters (where we have derived an optimal number of clusters as above), and then weight within these clusters
#
# The method was first outlined by Thomas Raffinot in the paper "*The Hierarchical Equal Risk Contribution Portfolio*".

# %% [markdown]
# ####  Step 1 - Cluster the assets

# %% [markdown]
# This works in a similar manner to the first step of HRP, although we use "ward" instead of "single" linkage.  The reasoning behind this is that under "single" linkage, only one two points in two different clusters need to be close for the clusters to merge, which can lead to a "chaining" effect. The "ward" method instead considers the distance between two clusters to be proportional to the distance between their centroids, which can be more robust to outliers.

# %%
# get the link matrix 

dist = ((1 - corr) / 2) ** 0.5
dist = pdist(dist, metric="euclidean")
link = sch.linkage(dist, "ward")

# %%
# plot the dendogram

fig, ax = plt.subplots()
fig.set_size_inches(20, 5)
ax.set_title("Dendrogram for clustering", fontsize=15)
sch.dendrogram(link, ax=ax, labels=data.columns);

# %% [markdown]
# #### Step 2 - Calculate optimal number of clusters

# %% [markdown]
# In order to calculate the optimal number of clusters, we calculate the gap statistic. The intuition behind this statistic is that if we have fewer than optimal clusters for our data and we add another, it should decrease the average distance of a point from the centroid of its cluster **by a greater amount** than would be expected if we had totally random data with no cluster structure. Conversely, as we clusters over and above the optimal amount, the decrease in average distance of a point from the centroid of its cluster **is lower than** we would expect for random data.
#
# To calculate the statistc, we thus work out the log of the squared average distance to a point's centroid for our data for k=1,2,3... clusters. We compare this to the expected values for random data (by averaging over a number of random samples), and pick the point at which the rate of decrease for our actual data falls behind that of the random data. 

# %%
actual_gaps, random_gaps, _ = get_gap_metric(returns_data.corr().values, n=15)

# %%
fig, ax = plt.subplots()
ind = list(range(1, actual_gaps.shape[0] + 1))
pd.Series(actual_gaps.squeeze(), index=ind).plot(kind="line", ax=ax, label="Actual data")
pd.Series(random_gaps.squeeze(), index=ind).plot(kind="line", ax=ax, label="Random data")
ax.legend()
ax.set_title("Actual vs Random avg errors")
fig.set_size_inches(15, 5)

# %% [markdown]
# Based on the below gap statistic, 3 or 6 clusters both look like reasonable choices.  I will use 6 here to make things a bit more interesting.

# %%
fig, ax = plt.subplots()
diff = random_gaps.squeeze() - actual_gaps.squeeze()
ind = list(range(1, actual_gaps.shape[0] + 1))
pd.Series(diff, index=ind).plot(kind="line", ax=ax)
ax.set_title("Gap statistic")
fig.set_size_inches(15, 5)

# %% [markdown]
# Below I show the 6 groups suggested by the algorithm.  Qualitatively, we can think of these as follows:
#
# 1) Rates assets 
#
# 2) Credit assets
#
# 3) Western equity markets
#
# 4) Asia-pac and EM equity markets
#
# 5) Inflation assets
#
# 6) Stress/diversifying assets

# %%
groupings = sch.fcluster(link, t=6, criterion="maxclust")

# %%
assets_by_group = {}
for j in range(1, 7):
    assets = [col for i, col in enumerate(data.columns) if groupings[i] == j]
    assets_by_group[j] = assets
    print(f"Assets in group {j}: {assets}")

# %%
fig, ax = plt.subplots()
reordered_assets = [data.columns[i] for i in sch.leaves_list(link)]
corr_matrix = returns_data.loc[:, reordered_assets].corr()
fig.set_size_inches(15, 10)
start_x = 0
start_y = 0
m = returns_data.shape[1]
for i in range(1, 7):
    n = len(assets_by_group[i])
    
    ax.axvline(x=start_x, ymin=(m - start_y - n) / m, ymax=(m - start_y) / m, color="blue", linewidth=3)
    ax.axvline(x=(start_x + n), ymin=(m - start_y - n) / m, ymax=(m - start_y) / m, color="blue", linewidth=3)
    ax.axhline(y=start_y, xmin=start_x / m, xmax=(start_x + n) / m, color="blue", linewidth=3)
    ax.axhline(y=(start_y + n), xmin=start_x / m, xmax=(start_x + n) / m, color="blue", linewidth=3)
    
    start_x = start_x + n
    start_y = start_y + n
    
sns.heatmap(corr_matrix, ax=ax)


# %%
weights = get_herc_weights(returns_data.loc[:, reordered_assets].cov(), assets_by_group)

# %%
fig, ax = plt.subplots()
weights.plot(kind="bar", ax=ax)
ax.yaxis.set_major_formatter(mtick.FuncFormatter("{:.0%}".format))
ax.set_title("HERC portfolio weights", fontsize=20)
fig.set_size_inches(15, 5)

# %%
