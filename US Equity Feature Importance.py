# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: machine-learning
#     language: python
#     name: machine-learning
# ---

# %% [markdown]
# # US Equity Feature Importance

# %% [markdown]
# The aim of this notebook is to explore some methods for feature importance - these can help to ascertain which variables are the most important to pay attention to, increase the transparency of the predictions made by machine learning models and help to remove extraneous variables from consideration. There are three main approaches available to us:
#
# 1) **Use explainable models** - so called "white-box" models are models simple enough that their output can be explained in terms of the input features solely by inspection of their coefficients and structure. Examples of this would be logistic regression models or single decision trees. Clearly, the downside here is that these models cannot capture highly non-linear relationships well, and may not be able to properly exploit very large datasets. However, the trade off may be worth it in situations where we have relatively small data sets and a limited number of predictive variables
#
# 2) **Alterations to input data** - sometimes, we can observe the impact on a fitted model's predictive power by altering our input variables. This allows us to see how reliant a model is on each variable, potentially in its out-of-sample predictions.
#
# 3) **Local model approximations** - more modern techniques like LIME and SHAP seek to fit "white-box" local approximations to more complicated models. This allows us to see what is driving the predictions made by our models at different points in our feature space. This is a more complex approach, and cannot provide global insights, but it allows us to work with more complicated models which may be able to better represent the highly non-linear relationships present in financial markets.
#
#
# The ideas herein lean heavily on material from the book "*Advances in Financial Machine Learning*" by Marcos Lopez de Prado.

# %% [markdown]
# ## Setting up

# %% [markdown]
# Here we import the required packages and the raw data that we require for our analysis.  I have taken the relevant data from Bloomberg or Reuters as appropriate.

# %%
# do imports

import random
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from finance_ml.feature_importance import get_mdi, get_permutation_importance, get_single_factor_importance, \
get_permutation_importance_in_sample
from finance_ml.cross_validation import get_purged_crossvalidation_score
from finance_ml.hp_tuning import fit_hyper_parameters
from finance_ml.cross_validation import PurgedKFold
import shap
import xgboost

plt.style.use("cardano")

# %% [markdown]
# Quick check for anything that looks odd...

# %%
# read in data

raw_data = pd.read_csv("raw_data/us_indicators.csv", index_col=0)
raw_data.index = pd.to_datetime(raw_data.index)
raw_data.describe().T

# %%
raw_data.isnull().sum()

# %% [markdown]
# Other than a couple of indices that start a bit later (Fed balance sheet and the Citi econ index), everything looks in order.

# %% [markdown]
# ## Feature engineering

# %% [markdown]
# Some of the features that we wish to look at are transformations of the raw data, which we apply here - in particular:
#
# * **200d_ma_z_score** - we convert the S&P 500 index a z-score by subtracting the 200 day moving average and dividing through by the 200d rolling standard deviation
#
# * **pc_ratio_ma_z_score** - we convert the CBOE put/call ratio index to a z-score by subtracting the 20 day moving average and dividing through by the 200d rolling standard deviation
#
# * **conf_board_90d_roc** - we take the % change in the US Conference Board leading indicator over its value 90d business days ago
#
# * **excess_liquidity_yoy** - we take the year on year change of the excess of the Fed balance sheet over outstanding treasuries
#
#
# We also construct a binary response variable based on whether the S&P went up or down over the subsequent 120 business days, which is roughly half a year.

# %%
# apply required transforms

features = raw_data.copy(deep=True)

# add 200d moving average z-score for s_and_p
features["200d_ma_z_score"] = features["s_and_p"] - features["s_and_p"].rolling(window=200).mean()
features["200d_ma_z_score"] /= features["s_and_p"].rolling(window=200).std()

# add 20d moving average z-score for call/put ratio
features["pc_ratio_ma_z_score"] = features["put_call_ratio"] - features["put_call_ratio"].rolling(window=5).mean()
features["pc_ratio_ma_z_score"] = features["put_call_ratio"].rolling(window=20).std()

# add 90d rate of change for conference board indicator
features["conf_board_90d_roc"] = features["conf_leading_indicator"] / features["conf_leading_indicator"].shift(90, freq="B") - 1

# add measure for US excess liquidity change
features["excess_liquidity_yoy"] = features["fed_balance_sheet"] - features["us_treasury_tob"]
features["excess_liquidity_yoy"] = features["excess_liquidity_yoy"] - features["excess_liquidity_yoy"].shift(260, "B")

features  = features[["spx_12m_fwd_pe", "200d_ma_z_score", "upgrades_to_downgrades", "pc_ratio_ma_z_score", "vix",
                      "citi_econ_surprise", "conf_board_90d_roc", "excess_liquidity_yoy"]]
features = features.dropna(how="any", axis=0)

# %%
# construct our response variable, and reindex the features

target = raw_data["s_and_p"].shift(-120, freq="B") / raw_data["s_and_p"] - 1
target = target.dropna()
target = pd.Series(np.where(target > 0, 1, 0), index=target.index)
features, target = features.align(target, join="inner", axis=0)

# %%
# construct index of periods for our target variable

periods = pd.Series(target.index + pd.offsets.BDay(120), index=target.index)

# %% [markdown]
# ## Data visualisation

# %% [markdown]
# As a quick check, we plot some basic data visualisations for our features. The below shows a correlation (based on daily values from 2003 to present) and a frequency plot showing the distribution of each feature.

# %%
sns.heatmap(features.corr());
fig = plt.gcf()
fig.set_size_inches(10, 10)
ax = plt.gca()
ax.set_title("Correlations", fontsize=20);

# %%
fig, axs = plt.subplots(4, 2)

for i, col in enumerate(features):
    
    r, c = i // 2, i % 2
    ax = axs[r, c]
    sns.histplot(features, x=col, ax=ax)
    title = " ".join([s.capitalize() for s in col.split("_")])
    ax.set_title(title)
    ax.set_xlabel("")

fig.set_size_inches(15, 15)
fig.tight_layout()

# %% [markdown]
# ## Fitting a model

# %% [markdown]
# ### Why do we need to fit a model?

# %% [markdown]
# Most feature importance methods are essentially a two step process:
#
# 1) fit a model to the data; and
#
# 2) interrogate the model to find out why it made certain predictions, or how its performance changes as a particular feature is changed
#
# Clearly, this requires us to fit a suitable model. In this notebook, I have tried five different models: a logistic regression model, support vector machine, neural network with a single hidden layer, a random forest classifier and a gradient boosting tree model classifier. 

# %% [markdown]
# ### How can we fit a model to our data?

# %% [markdown]
# To fit each, we need to be able to carry out some hyperparameter tuning. This is a non-trivial exercies on financial time series data, as we can't directly apply the standard apporach of k-fold cross validation due to the potential for data leakage between our test and training sets in each "fold".  Follow the approach of M. Lopez de Prado, for each "fold" we apply:
#
# * **embargoing** - we remove data after the test set where our features would be calculated from data which would lie in the test set
#
# * **purging** - we remove data before and after the test set where the period covered by the response variable (120 business days) would overlap between the two sets
#
# This approach is discussed in more detail [here](https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/). M. Lopez de Prado also suggests a procedure for weighting samples in classifiers where sampling is requried (eg random forest) to reduce the overlap between the response variable periods of selected data points. However, this is very time consuming to implement, and I have not done so here.  
#
# I have chosen to carry out 8-fold purged and embargoed cross-validation for each model, where I am using the area under the receiver operating curve to score the models.

# %%
# set parameters

splits = 8  # arbitratily, we use 8 splits
scoring_method = "roc_auc"  # use area under receiver operating curve

# %%
# logistic regression

pipe = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression())])
grid = {"classifier__C": [1e-2, 1e-1, 1, 10]}
log_reg_model = fit_hyper_parameters(features, target, periods, pipe, grid, n_splits=splits, scoring=scoring_method)
cv = PurgedKFold(periods, n_splits=splits, pct_embargo=0.1)
log_reg_cv = get_purged_crossvalidation_score(log_reg_model, features, target, cv, scoring_method=scoring_method)

# %%
# support vector machine

pipe = Pipeline([("scaler", StandardScaler()), ("classifier", SVC())])
grid = {"classifier__C": [1e-2, 1e-1, 1, 10], 
        "classifier__kernel": ["rbf", "poly", "sigmoid"], 
        "classifier__gamma": ["scale", "auto"], 
        "classifier__probability": [True]}
svm_model = fit_hyper_parameters(features, target, periods, pipe, grid, n_splits=splits, scoring=scoring_method)
cv = PurgedKFold(periods, n_splits=splits, pct_embargo=0.1)
svm_cv = get_purged_crossvalidation_score(svm_model, features, target, cv, scoring_method=scoring_method)

# %%
# neural network

pipe = Pipeline([("scaler", StandardScaler()), ("classifier", MLPClassifier())])
grid = {"classifier__hidden_layer_sizes": [32, 64, 128], 
        "classifier__activation": ["tanh", "relu"], 
        "classifier__alpha": [1e-4, 1e-3, 1e-2], 
        "classifier__learning_rate": ["constant"],
        "classifier__learning_rate_init": [1e-4, 1e-3, 1e-2, 1e-1]}
neural_net_model = fit_hyper_parameters(features, target, periods, pipe, grid, n_splits=splits, scoring=scoring_method)
cv = PurgedKFold(periods, n_splits=splits, pct_embargo=0.1)
neural_net_cv = get_purged_crossvalidation_score(neural_net_model, features, target, cv, scoring_method=scoring_method)

# %%
# random forest

pipe = Pipeline([("scaler", StandardScaler()), ("classifier", RandomForestClassifier())])
grid = {"classifier__n_estimators": [50, 100, 200], 
        "classifier__max_depth": [3, 6, 9, None], 
        "classifier__min_weight_fraction_leaf": [0, 0.1, 0.2], 
        "classifier__max_features": ["auto", "sqrt", "log2"]}
rf_model = fit_hyper_parameters(features, target, periods, pipe, grid, n_splits=splits, scoring=scoring_method)
cv = PurgedKFold(periods, n_splits=splits, pct_embargo=0.1)
rf_cv = get_purged_crossvalidation_score(rf_model, features, target, cv, scoring_method=scoring_method)

# %%
# gradient boosting 

pipe = Pipeline([("scaler", StandardScaler()), ("classifier", xgboost.XGBClassifier(use_label_encoder=False,
                                                                                    eval_metric="auc"))])
grid = {"classifier__gamma": [0, 0.1, 0.2, 0.3],
        "classifier__reg_alpha": [0, 1, 5],
        "classifier__reg_lambda": [0, 1, 5]}
xgb_model = fit_hyper_parameters(features, target, periods, pipe, grid, n_splits=splits, scoring=scoring_method)
cv = PurgedKFold(periods, n_splits=splits, pct_embargo=0.1)
xgb_cv = get_purged_crossvalidation_score(xgb_model, features, target, cv, scoring_method=scoring_method)

# %% [markdown]
# ### How do our models perform?

# %% [markdown]
# The results for the best model selected by our hyperparameter optimisation are show below, first each test period in isolation, then an average across all 8 time periods.  

# %%
all_cv = pd.concat([log_reg_cv, svm_cv, neural_net_cv, rf_cv, xgb_cv], axis=1, 
                   keys=["Logistic Regression", "Support Vector Machine", "Neural Network", 
                         "Random Forest", "Gradient Boosting Trees"])

all_cv

# %%
all_cv_avg = all_cv.mean()
all_cv_avg.name = "Average AUROC:"
all_cv_avg.to_frame().T

# %% [markdown]
# For the area under the receiver operating curve:
#
# * score >> 0.5 implies the model has good discriminative properties, and is better than just randomly guessing
# * score $\approx$ 0.5 implies the model has no real discriminative power, and is no better than randomly guessing
# * score << 0.5 implies that the model has learned the wrong relationships and is actually **worse** than randomly guessing
#
# All of our models are in the second and third camps. While this is probably not suprising (if you could reliably predict the S&P 500 by throwing a few daily series into some sklearn models, everyone would be very rich), it does mean that the my analysis is likely to be unreliable, as the feature importance methods discussed in the following section rely on the model they are applied on making sensible predictions.
#
# One observation that I made when looking at single feature models is that a lot of the relationships we would "expect" to see between the variables and the probability of the S&P going up over the next 120 bsuiness days have at times inverted violently over the period under consideration. In particular, during the GFC many of these relationships broke down - this means that when the GFC is one of the test sets, the model's training is distorted by this period and it performs poorly. However, when the GCF is instead in the test period, the model then learns the "expected" relationship and again cannot perform well on the test set either! 

# %% [markdown]
# ## Feature importance 

# %% [markdown]
# ### Mean decrease in impurity

# %% [markdown]
# The first measure is the so-called "mean decrease in impurity". This is an in-sample based measure wherein we fit a random forest classifier to our dataset, and then average the mean decrease in the measured impurity that is achieved in each tree when a given feature is used to split the data. Handily, this is the default measure that is uses in the sklearn implementation of random forest, so we don't need to write too much additional code.

# %%
mdi = get_mdi(features, rf_model["classifier"].estimators_)

# %%
mdi.sort_values().plot(kind="bar")
ax = plt.gca()
ax.set_title("MDI for US equity indicators", fontsize=20)
fig = plt.gcf()
fig.set_size_inches(15, 5);

# %% [markdown]
# ### Mean decrease in accuracy

# %% [markdown]
# A perhaps more sophisticated approach that we can try is "permutation importance", also known as mean decrease accuracy. Here, we first train a classifier on a set of training data, and measure its out of sample accuracy on the test data. Then, for each column in turn we permute the data, make another prediction and compare the accuracy of this prediction to our initial one. The idea is that permuting an informative features will damage performance greatly, while permuting an uninformative feature should make little difference.
#
# As this is is out of sample, we can do this in a (purged and embargoed) cross-validation manner.  We again use our random forest classifier, but this method is not tied to using tree-based models, and could easily work with any of the other three models too.

# %%
kfold_cv = PurgedKFold(periods=periods, n_splits=8, pct_embargo=0.1)
mda = get_permutation_importance(rf_model, features, target, kfold_cv, scoring_method="roc_auc_score")

# %%
mda.mean(axis=0).mul(100).sort_values().plot(kind="bar")
ax = plt.gca()
ax.set_title("Permutation loss US equity indicators", fontsize=20)
fig = plt.gcf()
fig.set_size_inches(15, 5);

# %% [markdown]
# The most obvious problem with both of the above is that they are based on a model which is not very predictive - this probably goes some way to explaining the lack of consistency between the two, and the fact that the MDA for all features is (in absolute terms) quite low.
#
# A more subtle issue with both MDI and MDA is that they suffer from "substitution effects" - if we have two closely related inputs then both will (misleadingly) appear redundant. To complement the above analysis, we would ideally supplement our analysis with some single-feature models. However, performance for such models is extremely poor (due to the reasons discussed earlier in the paper) and so I have not included this analysis.

# %% [markdown]
# ### SHAP

# %% [markdown]
# The final idea presented here is taken from [this](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3788037) paper. Shapley values are an idea from game theory, which aim to proportion the payoff of a game amongst its contributors. Here, we the "payoff" is the deviation of the predicted probability from our "average prediction" - by attempting to attribute this across our features, we can attempt to understand what is driving our models predictions.  Note that unlike "global" methods like MDI and MDA, this primarily aims to understand local behaviour of the model at different points.
#
# The way the method works is (broadly) that, for a given feature, we try to measure the impact of adding this feature to every other possible subgroup of features. Clearly, this is extremely computationally intensive, so in practice we restrict ourselves to a smaller class of model and then devise optimizations based on this limited class of models. TreeSHAP is one such approach, which restricts us to tree based models.  Happily this is implemented in the python SHAP package.

# %%
# get shap values

explainer = shap.TreeExplainer(xgb_model["classifier"])
shap_values = explainer(pd.DataFrame(rf_model["scaler"].transform(features), columns=features.columns))

# %%

rnd = random.randint(0, shap_values.shape[0] - 1) 
shap.plots.waterfall(shap_values[rnd])
date, actual_val = features.index[rnd], target.iloc[rnd]
print(f"SHAP plot for {date.date()}, actual value was {actual_val}")
