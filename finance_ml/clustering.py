from typing import Tuple, List, Optional, Dict
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from scipy.optimize import minimize


def get_gaps(corr_distance: np.array,
             n: Optional[int] = None) -> np.ndarray:

    distances = pdist(corr_distance, metric="euclidean")
    link_matrix = linkage(distances, metric="single")
    link_matrix = link_matrix.astype("int")

    if n is None:
        n = link_matrix[-1, -1]
    gaps = np.zeros(shape=(corr_distance.shape[0] - 1, 1))
    groups = {i: [i] for i in range(n)}
    costs = {i: 0 for i in range(n)}
    nxt_group = n

    for k in range(0, n - 1):
        # pick k,j to merge and remove them from current costs
        i, j = link_matrix[k, 0:2]
        group_i = groups.pop(i)
        cost_i = 2 * len(group_i) * costs.pop(i)
        group_j = groups.pop(j)
        cost_j = 2 * len(group_j) * costs.pop(j)

        # merge group and combine existing costs
        groups[nxt_group] = group_i + group_j
        costs[nxt_group] = cost_i + cost_j

        # now add additional cost of eac    h additional pair and normalise
        for i_0 in group_i:
            for j_0 in group_j:
                mx, mn = max(i_0, j_0), min(i_0, j_0)
                idx = n * mn + mx - ((mn + 2) * (mn + 1)) // 2
                costs[nxt_group] += distances[idx]  # L2 norm
        costs[nxt_group] /= (2 * len(groups[nxt_group]))

        # store the total cost
        gaps[len(groups) - 1, 0] = costs[nxt_group]
        nxt_group += 1

    return np.log(gaps)


def get_gap_metric(corr: np.ndarray,
                   n: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # get actual gaps for our correlation matrix
    actual_gaps = get_gaps(np.sqrt(abs(1 - corr) / 2))

    # now get expected gaps for random data
    all_random_gaps = np.zeros(shape=(corr.shape[0] - 1, n))
    rng = np.random.default_rng()
    for i in range(n):
        random_vals = rng.uniform(low=0, high=1, size=corr.shape)
        all_random_gaps[:, i:i + 1] = get_gaps(random_vals)

    random_gaps = all_random_gaps.mean(axis=1, keepdims=True)
    std = np.sqrt(1 + 1 / n) * all_random_gaps.std(axis=1, keepdims=True)
    return actual_gaps, random_gaps, std


def get_clustered_order(link_matrix: np.ndarray) -> np.ndarray:
    """
    return a 1D array of our assets 0 to n ordered so
    that similar assets (as determined by the link matrix) are
    clustered together
    """

    # create link matrix and stub of sorted index
    link_matrix = link_matrix.copy()
    link_matrix = link_matrix.astype("int")
    clustered_order = pd.Series(link_matrix[-1, 0:2])
    number_items = link_matrix[-1, -1]

    # iterate over link matrix to unpack clusters
    while clustered_order.max() >= number_items:
        # reset index, using 2* to make space for unpacking
        clustered_order.index = range(0, clustered_order.shape[0] * 2, 2)
        clusters = clustered_order[clustered_order >= number_items]

        # get indices of the current clusters of size > 1, and the children of those clusters
        cluster_indices = clusters.index
        child_cluster_indices = clusters.values - number_items

        # unpack the clusters and recombine
        clustered_order[cluster_indices] = link_matrix[child_cluster_indices, 0]
        child_clusters = pd.Series(link_matrix[child_cluster_indices, 1], index=cluster_indices + 1)
        clustered_order = clustered_order.append(child_clusters)
        clustered_order = clustered_order.sort_index()

    return clustered_order.values


def get_quasi_diagonal_matrices(data: pd.DataFrame,
                                link_method="single") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    create a covariance and correlation matrix from the input returns data
    where we have clustered together similar assets (based on the realised
    correlation structure)
    """

    corr = np.corrcoef(data.values, rowvar=False)
    dist = ((1 - corr) / 2) ** 0.5
    dist = pdist(dist, metric="euclidean")
    link = linkage(dist, link_method)
    clustered_order = get_clustered_order(link)
    corr_diag = pd.DataFrame(data.iloc[:, clustered_order].corr())
    cov_diag = pd.DataFrame(data.iloc[:, clustered_order].cov())
    return corr_diag, cov_diag


def get_hrp_weights(returns_data: pd.DataFrame,
                    link_method="single") -> pd.Series:
    """
    get a series of weights based on HRP allocation
    a la M. Lopez de Prado
    """

    def _get_cluster_var(cov: pd.DataFrame,
                         items: List[str]) -> float:
        """
        get var of risk parity holding in the given
        items
        """

        cov_subset = cov.loc[items, items]
        cov_diag = pd.Series(np.diag(cov_subset.values), index=items)
        w = 1 / cov_diag
        w_norm = w / w.sum()
        return w_norm.T @ cov_subset @ w_norm

    _, cov_matrix = get_quasi_diagonal_matrices(returns_data, link_method)
    weights = pd.Series(1, index=cov_matrix.index)
    all_sub_groups = [list(cov_matrix.columns)]

    while len(all_sub_groups):
        # bisect each group
        bisected_sub_groups = []
        for sub_group in all_sub_groups:
            if len(sub_group) > 1:
                n = len(sub_group) // 2
                bisected_sub_groups.extend([sub_group[:n], sub_group[n:]])

        all_sub_groups = bisected_sub_groups

        # now re-weight based on each sub-group
        for i in range(0, len(all_sub_groups), 2):  # step size 2 as process in pairs
            grp_1, grp_2 = all_sub_groups[i], all_sub_groups[i + 1]
            grp_1_var = _get_cluster_var(cov_matrix, grp_1)
            grp_2_var = _get_cluster_var(cov_matrix, grp_2)
            alpha = 1 - grp_1_var / (grp_1_var + grp_2_var)
            weights[grp_1] *= alpha
            weights[grp_2] *= (1 - alpha)

    return weights


def get_equal_risk_contribution_weights(cov: pd.DataFrame,
                                        scaling: int = 1000) -> pd.Series:
    """
    get weights using standard equal-contribution-to-risk
    risk parity portfolio construction
    """

    rng = np.random.default_rng()
    weights = rng.uniform(low=0, high=1, size=cov.shape[1])
    weights /= weights.sum()

    def obj(w: np.array) -> float:

        w = w.reshape(-1, 1)
        numerator = (w.T @ cov.values @ w).squeeze()
        denominator = cov.values @ w * w.shape[0]
        sum_square = 0
        for i in range(w.shape[0]):
            sum_square += np.power(w[i, 0] - numerator / denominator[i, 0], 2)
        return scaling * sum_square

    # noinspection PyTypeChecker
    opt = minimize(fun=obj,
                   x0=weights,
                   method="SLSQP",
                   options=dict(disp=False),
                   constraints=[dict(type="eq", fun=lambda x: sum(x) - 1),
                                dict(type="ineq", fun=lambda x: x)],
                   bounds=[(0, 1) for _ in range(len(weights))]
                   )

    if not opt.success:
        raise ValueError("optimisation failed")

    return pd.Series(data=opt.x, index=cov.index)


def get_global_minimum_weights(cov: pd.DataFrame,
                               scaling: int = 1000) -> pd.Series:
    """
    get weights corresponding to the global min vol portfolio
    based on the given covariance matrix
    """

    rng = np.random.default_rng()
    weights = rng.uniform(low=0, high=1, size=cov.shape[0])
    weights /= sum(weights)

    def obj(w: np.ndarray) -> float:
        w = w.reshape(-1, 1)
        return scaling * (w.T @ cov.values @ w).squeeze()  # scale to yearly so is big enough for optimization

    # noinspection PyTypeChecker
    opt = minimize(fun=obj,
                   x0=weights,
                   method="SLSQP",
                   options=dict(disp=False),
                   constraints=[dict(type="eq", fun=lambda x: np.sum(x) - 1)],
                   bounds=[(0, 1) for _ in range(len(weights))]
                   )

    if not opt.success:
        raise ValueError("optimisation failed")

    return pd.Series(data=opt.x, index=cov.index)


def get_herc_weights(cov_matrix: pd.DataFrame,
                     assets_by_group: Dict[int, List[str]]) -> pd.Series:
    """
    get portfolio weight's using Raffinot's HERC algorithm
    """

    # step 1 - weight the clusters using bisection
    all_subgroups = [list(assets_by_group)]
    weights = pd.Series(1 / cov_matrix.shape[0], index=cov_matrix.columns)

    def _get_cluster_vol(assets):

        cov = cov_matrix.loc[assets, assets]
        inv_vols = 1 / np.diag(cov)
        inv_vols /= inv_vols.sum()
        inv_vols = inv_vols.reshape(-1, 1)
        cluster_vol = np.sqrt(inv_vols.T @ cov @ inv_vols)
        return cluster_vol.squeeze()

    while all_subgroups:

        bisected_sub_groups = []
        for sub_group in all_subgroups:
            size = len(sub_group)
            if size == 1:
                continue
            grp_1, grp_2 = sub_group[:size // 2], sub_group[size // 2:]
            bisected_sub_groups.extend([grp_1, grp_2])

        all_subgroups = bisected_sub_groups
        for i in range(0, len(all_subgroups), 2):
            grp_1, grp_2 = all_subgroups[i], all_subgroups[i + 1]
            assets_1, assets_2 = [], []
            for j in grp_1:
                assets_1.extend(assets_by_group[j])
            for j in grp_2:
                assets_2.extend(assets_by_group[j])
            vol_1 = _get_cluster_vol(assets_1)
            vol_2 = _get_cluster_vol(assets_2)
            alpha = vol_1 / (vol_1 + vol_2)
            total_weight = weights[assets_1 + assets_2].sum()
            weights[assets_1] = (1 - alpha) * total_weight / len(assets_1)
            weights[assets_2] = alpha * total_weight / len(assets_2)

    return weights
