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
# # Feature Importance

# %% [markdown]
# Backtesting is not a good way of conducting research - instead, we should look at feature importance measures. This notebook sets out some methods for doing this - I havce used the UCI heart disease data set as an example.

# %%
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from typing import List, Tuple
from matplotlib import pyplot as plt, ticker as mtick
from scipy.stats import weightedtau

# %%
data = pd.read_csv("raw_data/heart.csv")

# %%
data.describe().T

# %%
data.sample(5)

# %%
features, target = data.drop("target", axis=1), data["target"]


# %% [markdown]
# ## Mean Decrease Impurity

# %% [markdown]
# The "*Mean Decrease Impurity* ("MDI") measures the decrease in impurity that results from a tree splitting on a given feature. The higher this decrease, the more useful the feature is. Note that:
#
# 1) this is an in sample measure; 
#
# 2) the MDI only works with random forest classifiers; and
#
# 3) this method does not address substitution effects, in that two correlated (but useful) variables will appear less useful.
#
# The latter point is because if we have two related features A and B, then whenever we have a tree where we have split by A, if we then split by B the gain will be lower (as most of the useful discriminative power of the feature has already been provided) which drags the mean down over all trees.
#
# Helpfully, the MDI is the defactor feature importance measure that sklearn uses for its RandomForestClassifier (and regressor) classes, so we don't have to implement anything ourselves. Also, as it can be calculated on the fly, it is pretty quick.

# %%
def get_mdi(fitted_trees: List[DecisionTreeClassifier],
            feature_names: List[str]) -> pd.DataFrame:
    """
    get the mean decrease impurity from our fitted
    decision trees
    """
    
    importance = {i: tree.feature_importances_ for i, tree in enumerate(fitted_trees)}
    importance = pd.DataFrame.from_dict(importance, orient="index")
    importance.columns = feature_names
    importance = importance.replace(0, np.NaN)
    mdi = pd.concat([importance.mean(), importance.std()], keys=["mean", "std"], axis=1)
    mdi /= mdi["mean"].sum()
    return mdi


# %%
rf = RandomForestClassifier(n_estimators=200)
rf.fit(features, target)
mdi = get_mdi(rf.estimators_, features.columns)
mdi["mean"].sort_values().plot(kind="bar")
ax = plt.gca()
ax.yaxis.set_major_formatter(mtick.FuncFormatter("{:.0%}".format));
ax.set_title("MDI Feature Importance")
fig = plt.gcf()
fig.set_size_inches(15, 5)


# %% [markdown]
# ## Mean Decrease Accuracy

# %% [markdown]
# To calculate the *Mean Decrease Accuracy* ("MDA"), we train a classifier and compute its out of sample performance, then permute a feature and recalculate the out of sample performance with the feature permuted. The idea here is that permuting a useful variable is hugely damaging to a model's performance, while permuting a largely useless variable should have little impact. Note that:
#
# 1) this is an out of sample measure;
#
# 2) this works for arbitrary models and performance metrics, not just random forest and entropy; and
#
# 3) like MDI, this suffers from issues with substition effects.
#
# The latter point is because if we have two highly related variables A and B, permuting A will likely have little impact as long as B is still in order. As such, both will appear to be of little use, even if in actual fact both have useful predicitve power.
#
# MDA is implemented by the random forest classifier, but below we "roll our own" so that we can use this with other models (the example here uses naive bayes with an F1 score accuracy measure). The combo of this and the fact that it is out of bag does mean that this is a bit more complex than MDI from a practical perspective.

# %%
def get_mda(features: pd.DataFrame,
            target: pd.Series,
            folds: int) -> pd.DataFrame:
    """
    compute the mean decrease accuracy for 
    our features using naive bayes and f1 score
    """
    
    kf = KFold(n_splits=folds, shuffle=True)
    out = {f: [] for f in features.columns}
    for train_index, test_index in kf.split(features):
        train_features, train_target = features.iloc[train_index, :], target.iloc[train_index]
        test_features, test_target = features.iloc[test_index, :], target.iloc[test_index]
        
        classifier = GaussianNB()
        classifier.fit(train_features, train_target)
        pred = classifier.predict(test_features)
        oos_score = f1_score(pred, test_target)
        for col in features:
            permuted = test_features.copy(deep=True)
            np.random.shuffle(permuted[col].values)
            pred_permuted = classifier.predict(permuted)
            permuted_score = f1_score(pred_permuted, test_target)
            out[col].append(oos_score - permuted_score)
        
    mda = pd.DataFrame(out)
    return mda
        


# %%
get_mda(features, target, 6).mean().sort_values().plot(kind="bar")
ax = plt.gca()
ax.yaxis.set_major_formatter(mtick.FuncFormatter("{:.0%}".format))
ax.set_title("MDA Feature Importance")
fig = plt.gcf()
fig.set_size_inches(15, 5)


# %% [markdown]
# ## Single Feature Importance

# %% [markdown]
# An issue with both MDI and MDA is that they struggle to capture the importance of features which are highly related to other features in our data (substitution effect). To augment the above, we can also look at *Sinlge Feature Importance* ("SFI") to overcome this. Here we just look at out of sample performance of each feature in isolation. Note that:
#
# 1) this is an out of sample measure;
#
# 2) again, we can use arbitrary models for this; and
#
# 3) as we are only considering one feature at a time, there are no substitution effects
#
#
# Again we use a naive bayes model with the F1 score.

# %%
def get_sfi(features: pd.DataFrame,
            target: pd.Series,
            folds: int) -> pd.DataFrame:
    """
    get the single feature importance for
    each feature
    """
    
    kf = KFold(n_splits=folds, shuffle=True)
    out = {f: [] for f in features.columns}
    
    for train_index, test_index in kf.split(features):
        train_features, train_target = features.iloc[train_index, :], target.iloc[train_index]
        test_features, test_target = features.iloc[test_index, :], target.iloc[test_index]
        
        for col in features:
            cf = GaussianNB()
            cf.fit(train_features[col].values.reshape(-1, 1), train_target)
            pred = cf.predict(test_features[col].values.reshape(-1, 1))
            out[col].append(f1_score(pred, test_target))
    
    sfi = pd.DataFrame(out)
    return sfi


# %%
get_sfi(features, target, 5).mean().sort_values().plot(kind="bar")
ax = plt.gca()
ax.yaxis.set_major_formatter(mtick.FuncFormatter("{:.0%}".format))
ax.set_title("Single Feature Importance")
fig = plt.gcf()
fig.set_size_inches(15, 5)


# %% [markdown]
# ## Principal Component Analysis

# %% [markdown]
# To get more comfort that our data contains genuinely useful information for predicting our target, and we are not just fitting to noise, we can calculate the principal components of our data and then plot the eigenvalues of our data against the MDI based on a random forest model trained on these features. If the data has predictive power, we would hope to see that the principal components with the largest eigenvalues (ie those explaining the greatest % of variance in the data) with the greatest MDI. 
#
# To make this more concrete, we calculate the **weighted** Kendall's Tau measure between the eigenvalue size and the MDI. The Kendall's Tau statistic measures the difference between the number of concordant pairs(ie same order in both data sets) and the number of discordant pairs (ie the reverse order between the two data sets), as a proportion of the total number of pairs.  We use a weighted version of this to penalise different orderings of the most significant principal components more than the less significant ones.

# %%
def get_pca(features_standard: pd.DataFrame,
            var_threshold: float = 0.95) -> Tuple[pd.Series, pd.DataFrame]:
    """
    get principle components
    """
    
    # get our eigenvalues and sort in order of size
    eig_val, eig_vec = np.linalg.eigh(features_standard.cov())
    idx = eig_val.argsort()[::-1]
    eig_val, eig_vec = eig_val[idx], eig_vec[:, idx]

    # wrap as pandas objects
    eig_val = pd.Series(eig_val, index=[f"pc_{1 + i}" for i in range(eig_val.shape[0])], name="eig_vals")
    eig_vec = pd.DataFrame(eig_vec, index=features_standard.columns, columns=eig_val.index)
    eig_vec = eig_vec.loc[:, eig_val.index]

    # reduce dimensions by getting rid of small eigenvalues
    cumulative_var = eig_val.cumsum()/eig_val.sum()
    cutoff = cumulative_var.values.searchsorted(var_threshold)
    eig_val, eig_vec = eig_val.iloc[:cutoff + 1], eig_vec.iloc[:, :cutoff + 1]
    return eig_val, eig_vec


# %%
# calculate the features in the PCA basis

features_standard = (features - features.mean())/features.std()
eig_vals, components = get_pca(features_standard)
features_pca = features @ components  # rewrite our features in the PCA basis

# %%
# fit random forest model to the PCA features

rf = RandomForestClassifier(n_estimators=200)
rf.fit(features_pca, target)
mdi = get_mdi(rf.estimators_, features_pca.columns)
mdi = mdi.join(eig_vals)
mdi[["mean", "eig_vals"]].plot(kind="scatter", y="mean", x="eig_vals");
ax = plt.gca()
ax.set_title("Plot of eigenvalues vs. mean MDI")
ax.yaxis.set_major_formatter(mtick.FuncFormatter("{:.0%}".format))
fig = plt.gcf()
fig.set_size_inches(15, 5)

# %%
weightedtau(mdi["mean"], mdi["eig_vals"]).correlation

# %% [markdown]
# Correlation does not look great - I think this might be because we have relatively few features, and also because using the binary variables (eg sex) in this analysis might give weird results. I should look at this in more detail.

# %% [markdown]
# ## Testing on synthetic dataset

# %% [markdown]
# To conclude the notebook, we create some synthetic data and show how the above methods perform.

# %%
from sklearn.datasets import make_classification


# %%
def get_synthetic_data(total_features: int = 40,
                       informative_features: int = 10,
                       redundanat_features: int = 10,
                       samples: int = 10000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    generate a synthetic data set
    """
    
    
    features, target = make_classification(n_samples=samples, 
                                      n_features=total_features,
                                      n_informative=informative_features,
                                      n_redundant=redundant_features,
                                      random_state=0,
                                      shuffle=False)

    times = pd.date_range(periods=samples, freq=pd.tseries.offsets.BDay(), end=datetime.today())
    features = pd.DataFrame(features, index=times)
    target = pd.Series(target, index=times, name="target")
    
    cols = [f"inf_{i + 1}" for i in range(informative_features)]
    cols += [f"red_{i + 1}" for i in range(redundant_features)]
    cols += [f"ns_{i + 1}" for i in range(total_features - informative_features - redundant_features)]
    features.columns = cols
    
    return features, target

synth_features, synth_target = get_synthetic_data()

# %%
# fit a random forest model

rf = RandomForestClassifier(n_estimators=200)
rf.fit(synth_features, synth_target)
mdi = get_mdi(rf.estimators_, synth_features.columns)
ax = plt.gca()
ax.yaxis.set_major_formatter(mtick.FuncFormatter("{:.0%}".format))
ax.set_title("MDI")
ax.axhline(1 / 40, color="red", linestyle="--")
fig = plt.gcf()
fig.set_size_inches(15, 5);
mdi["mean"].sort_values(ascending=False).plot(kind="bar", ax=ax);

# %%
mda = get_mda(synth_features, synth_target, 5)
ax = plt.gca()
ax.set_title("MDA")
ax.yaxis.set_major_formatter(mtick.FuncFormatter("{:.0%}".format))
fig = plt.gcf()
fig.set_size_inches(15, 5)
mda.mean().sort_values(ascending=False).plot(kind="bar");

# %%
sfi = get_sfi(synth_features, synth_target, 5)
ax = plt.gca()
ax.set_title("SFI")
ax.yaxis.set_major_formatter(mtick.FuncFormatter("{:.0%}".format))
fig = plt.gcf()
fig.set_size_inches(15, 5)
sfi.mean().sort_values(ascending=False).plot(kind="bar");
