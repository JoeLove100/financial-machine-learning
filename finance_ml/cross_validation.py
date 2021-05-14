import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Union, Optional, Tuple
from sklearn.model_selection._split import _BaseKFold  # need this to create derived implementation
from sklearn.metrics import log_loss, accuracy_score, f1_score, roc_auc_score


VALID_SCORING_METHODS = ["neg_log_loss", "accuracy", "f1", "roc_auc"]


def get_embargo_times(times: pd.Series,
                      pct_embargo: float) -> pd.Series:
    """
    for each time, get the embargo value such that data leakage
    from features after the test period is prevented

    :param times: a series of all times in the data set
    :param pct_embargo: the % of the total dataset that the embargo after
    the training period should represent
    :return embargo_times: a series where the index gives the test end
    date and the value is the corresponding embargoed end date
    """

    step_size = int(times.shape[0] * pct_embargo)
    if step_size == 0:
        embargo_times = pd.Series(times.values, index=times)  # no embargo required
    else:
        embargo_times = pd.Series(times.iloc[step_size:].values, index=times.iloc[:-step_size])
        embargo_times = embargo_times.append(pd.Series(times.iloc[-1], index=times.iloc[-step_size:]))

    return embargo_times


def get_purged_train_times(periods: pd.Series,
                           test_start_dates: List[datetime],
                           test_end_dates: List[datetime]) -> pd.Series:
    """
    remove all times from the full set of periods which overlap
    with the selected test dates

    :param periods: a series where the index gives the start of each target
    return period, and the value gives the end of the period
    :param test_start_dates: the first dates of the test periods
    :param test_end_dates: the final dates of the test periods
    :return: purged_periods: periods with entries overlapping the test set
    removed
    """

    if len(test_end_dates) != len(test_start_dates):
        raise ValueError("Need to provide as many start dates as end dates")

    purged_periods = periods.copy(deep=True)
    for start, end in zip(test_start_dates, test_end_dates):
        starts_in_test = periods[(start <= periods.index) & (periods.index <= end)].index
        ends_in_test = periods[(start <= periods) & (periods <= end)].index
        envelops_test = periods[(periods.index <= start) & (periods >= end)].index
        purged_periods = purged_periods.drop(starts_in_test.union(ends_in_test).union(envelops_test))

    return purged_periods


def get_single_test_train_period(periods: pd.Series,
                                 start: datetime,
                                 end: datetime,
                                 embargo_pct: float) -> Tuple[pd.Series, pd.Series]:
    """
    get a purged and embargoed set of times for a single
    test set training period
    """

    embargo_times = get_embargo_times(pd.Series(periods.index), embargo_pct)
    purged_train_times = get_purged_train_times(periods, [start], [embargo_times.loc[end]])
    purged_test_times = periods.loc[start: end]
    return purged_train_times, purged_test_times


class PurgedKFold(_BaseKFold):
    """
    Following the suggestion of M. Lopez de Prado, we create a class
    derived from sklearn's KFold base class in order to implement embargo
    and purging
    """

    def __init__(self,
                 periods: pd.Series,
                 n_splits: int = 3,
                 pct_embargo: float = 0):

        super(PurgedKFold, self).__init__(n_splits=n_splits,
                                          shuffle=False,  # assume data is continuous time series
                                          random_state=None)
        if not isinstance(periods, pd.Series):
            raise ValueError("Periods must be provided as pandas series")

        self._periods = periods
        self._pct_embargo = pct_embargo

    def split(self,
              x: pd.DataFrame,
              y: Optional[pd.Series] = None,
              groups=None):

        if (x.index == self._periods.index).sum() != len(self._periods):
            raise ValueError("X and target periods must have same index")

        indices = np.arange(x.shape[0])
        embargo_size = int(x.shape[0] * self._pct_embargo)
        test_periods = [(arr[0], arr[-1] + 1) for arr in np.array_split(indices.copy(), self.n_splits)]

        for start_index, end_index in test_periods:
            start_date = self._periods.index[start_index]
            max_end_date = max(self._periods.iloc[start_index: end_index])
            max_end_index = self._periods.index.searchsorted(max_end_date)

            train_dates_lower = self._periods[self._periods <= start_date].index
            train_indices_pre_test = self._periods.index.searchsorted(train_dates_lower)
            train_dates_upper = self._periods.index[max_end_index + embargo_size + 1:]
            train_indices_post_test = self._periods.index.searchsorted(train_dates_upper)

            train_indices = np.concatenate([train_indices_pre_test, train_indices_post_test])
            test_indices = indices[start_index: end_index]
            yield train_indices, test_indices


def _get_score(clf,
               x_test: np.ndarray,
               y_test: np.ndarray,
               scoring_method: str) -> float:
    """
    get the score based on the given scoring method
    """

    if scoring_method in ["accuracy", "f1"]:
        pred = clf.predict(X=x_test)
        if scoring_method == "accuracy":
            return accuracy_score(y_test, pred)
        else:
            return f1_score(y_test, pred)
    else:
        prob = clf.predict_proba(X=x_test)[:, 1]  # get prob of class 1 only
        if scoring_method == "neg_log_loss":
            return -log_loss(y_test, prob)
        else:
            return roc_auc_score(y_test, prob)


def get_purged_crossvalidation_score(clf,
                                     x: pd.DataFrame,
                                     y: pd.Series,
                                     cross_val: PurgedKFold,
                                     scoring_method: str = "neg_log_loss") -> pd.Series:
    """
    use our purged k-fold cross validation class to compute a
    cross validation score
    """

    if scoring_method not in VALID_SCORING_METHODS:
        raise ValueError(f"Scoring method {scoring_method} not recognized - valid methods are "
                         f"{', '.join(VALID_SCORING_METHODS)}")

    score = pd.Series(dtype=float)
    for train, test in cross_val.split(x):
        cls_fitted = clf.fit(X=x.iloc[train, :],
                             y=y.iloc[train])
        test_start_date = x.index[min(test)].date()
        test_end_date = x.index[max(test)].date()
        period = f"{test_start_date} - {test_end_date}"
        score.loc[period] = _get_score(cls_fitted, x.iloc[test, :], y.iloc[test], scoring_method)

    return pd.Series(score)


if __name__ == "__main__":

    from sklearn.linear_model import LogisticRegression

    test_times = pd.Series(pd.date_range(datetime(2019, 1, 1), datetime(2021, 1, 31), freq="B"))
    test_all_periods = pd.Series((test_times + pd.offsets.BDay(60)).values, index=test_times)
    test_start, test_end = datetime(2020, 6, 1), datetime(2020, 9, 30)

    rng = np.random.default_rng(1234)
    test_x = pd.DataFrame(rng.normal(size=(len(test_times), 5)), index=test_times)
    test_y = pd.Series(rng.choice([0, 1], size=(len(test_times),), replace=True), index=test_times)

    test_cv = PurgedKFold(test_all_periods, pct_embargo=0.1)
    classifier = LogisticRegression()
    get_purged_crossvalidation_score(classifier, x=test_x, y=test_y, cross_val=test_cv)
