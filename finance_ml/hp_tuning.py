import pandas as pd
from typing import Union, Dict, List, Any
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from finance_ml.cross_validation import PurgedKFold, VALID_SCORING_METHODS


def fit_hyper_parameters(features: pd.DataFrame,
                         target: pd.Series,
                         periods: pd.Series,
                         classifier_pipeline: Pipeline,
                         parameter_grid: Dict[str, List[Any]],
                         scoring: str,
                         n_splits: int = 3,
                         pct_embargo: float = 0,
                         grid_search: bool = True,
                         n_jobs: int = -1,
                         **fit_params):
    """
    tune classifier hyper-parameters through k-fold
    cross validation which has been purged and embargoed
    to prevent data leakage

    :param features:
    :param target: pandas series of labels for data
    :param periods:
    :param classifier_pipeline:
    :param parameter_grid:
    :param scoring: method to assess performance of a fitted classifier by - can be one of
    accuracy, negative log loss, F1 score or area under the receiver operating curve
    :param n_splits: number of folds to use in k-fold validation
    :param pct_embargo: % size of embargo to be applied after test set in cross validation
    :param grid_search: boolean flag for whether to apply a grid search or a random search
    :param n_jobs: number of jobs to run in parallel during the cross-validation grid search, defaulting
    to -1 (ie use all available processors)

    :return: returns the best model found through our purged and embargoed cross validation procedure
    """

    if scoring not in VALID_SCORING_METHODS:
        raise ValueError(f"Scoring method {scoring} not recognized. Valid scoring methods are"
                         f"{', '.join(VALID_SCORING_METHODS)}")

    inner_cv = PurgedKFold(n_splits=n_splits, periods=periods, pct_embargo=pct_embargo)

    if grid_search:
        search = GridSearchCV(estimator=classifier_pipeline, param_grid=parameter_grid, scoring=scoring,
                              cv=inner_cv, n_jobs=n_jobs)
    else:
        search = RandomizedSearchCV(estimator=classifier_pipeline, param_distributions=parameter_grid,
                                    scoring=scoring, cv=inner_cv, n_jobs=n_jobs, iid=False)

    best_model = search.fit(features, target, **fit_params).best_estimator_
    return best_model
