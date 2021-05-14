import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, f1_score, roc_auc_score
from finance_ml.cross_validation import PurgedKFold


def get_mdi(features: pd.DataFrame,
            fitted_trees: pd.Series) -> pd.Series:
    """
    fit a random forest classifier to the data and use this
    to compute the 'Mean Decrease in Impurity'
    """

    importances = {i: tree.feature_importances_ for     i, tree in enumerate(fitted_trees)}
    importances = pd.DataFrame.from_dict(importances, orient="index")
    importances.columns = features.columns
    importances = importances.replace(0, np.NaN)  # as we have set max features to 1

    mdi = importances.mean(axis=0)  
    return mdi


def get_permutation_importance(classifier,
                               x: pd.DataFrame,
                               y: pd.Series,
                               cv: PurgedKFold,
                               scoring_method="neg_log_loss") -> pd.DataFrame:
    """
    fit a random forest classifier to the part of the data
    """

    if scoring_method not in ["neg_log_loss", "roc_auc_score"]:
        raise ValueError(f"Scoring method {scoring_method} is not recognized")

    scores, permuted_scores = pd.Series(), pd.DataFrame(columns=x.columns)
    for i, (train, test) in enumerate(cv.split(x)):
        x_train, y_train = x.iloc[train, :], y.iloc[train]
        x_test, y_test = x.iloc[test, :], y.iloc[test]
        fit = classifier.fit(x_train, y_train)
        if scoring_method == "neg_log_loss":
            prob = fit.predict_proba(x_test)
            score_ = -log_loss(y_test, prob)
        else:
            pred = fit.predict(x_test)
            score_ = f1_score(y_test, pred)

        scores.loc[i] = score_

        for col in x.columns:
            x_test_perm = x_test.copy(deep=True)
            np.random.shuffle(x_test_perm[col].values)
            if scoring_method == "neg_log_loss":
                prob = classifier.predict_proba(x_test_perm)
                score_ = -log_loss(y_test, prob)
            else:
                pred = classifier.predict(x_test_perm)
                score_ = f1_score(y_test, pred)

            permuted_scores.loc[i, col] = score_

    importance = permuted_scores.subtract(scores, axis=0) * -1
    return importance


def get_permutation_importance_in_sample(classifier,
                                         x: pd.DataFrame,
                                         y: pd.Series,
                                         scoring_method="neg_log_loss") -> pd.Series:
    """
    get permutation feature importance on the entire data set
    (ie in sample) rather than through cross validation
    """

    if scoring_method not in ["neg_log_loss", "roc_auc_score"]:
        raise ValueError(f"Scoring method {scoring_method} is not recognized")

    permuted_scores = pd.Series(dtype=float)
    clf = classifier.fit(x.values, y.values)
    if scoring_method == "neg_log_loss":
        prob = clf.predict_proba(x.values)
        score = - log_loss(y.values, prob[:, 1])
    else:
        prob = clf.predict_proba(x.values)
        score = roc_auc_score(y.values, prob[:, 1])

    for col in x:
        permuted_x = x.copy(deep=True)
        np.random.shuffle(permuted_x[col].values)
        if scoring_method == "neg_log_loss":
            prob = clf.predict_proba(x.values)
            perm_score = - log_loss(y.values, prob[:, 1])
        else:
            prob = clf.predict_proba(x.values)
            perm_score = roc_auc_score(y.values, prob[:, 1])

        permuted_scores.loc[col] = score - perm_score

    return permuted_scores


def get_single_factor_importance(classifier,
                                 x: pd.DataFrame,
                                 y  : pd.Series,
                                 cv: PurgedKFold,
                                 scoring_method: str = "neg_log_loss"):

    if scoring_method not in ["neg_log_loss", "f1_score"]:
        raise ValueError(f"Scoring method {scoring_method} not recognized")

    coefs = pd.Series()
    scores = pd.Series()
    for i, (train, test) in enumerate(cv.split(x, y)):
        x_train, y_train = x.iloc[train, :], y.iloc[train]
        x_test, y_test = x.iloc[test, :], y.iloc[test]
        fitted_classifier = classifier.fit(x_train, y_train)
        if scoring_method == "neg_log_loss":
            prob = fitted_classifier.predict_proba(x_test)
            score_ = -log_loss(y_test, prob)
        else:
            pred = fitted_classifier.predict_proba(x_test)
            score_ = roc_auc_score(y_test, pred[:, 1])

        test_start = y.index[min(test)].date()
        test_end = y.index[max(test)].date()
        scores.loc[f"{test_start} to {test_end}"] = score_
        coefs.loc[f"{test_start} to {test_end}"] = fitted_classifier.coef_[0][0]

    return scores

