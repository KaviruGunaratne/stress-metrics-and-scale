"""Module providing functions to calculate the Normalized Stress score, 
scale the Stress score within the "interesting" range, find the minimum Stress value, 
and find the scalar value that corresponds to the minimum Stress value for each 
dimensionality reduction technique."""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.isotonic import IsotonicRegression
from zadu.measures import *


def compute_stress_kruskal(D_high, D_low):
    """
    Computes the non-metric stress between high dimensional distances D_high
    and low dimensional distances D_low. Invariant to scale of D_low.
    """
    dij = D_high
    xij = D_low

    # Find the indices of dij that when reordered, would sort it. Apply to both arrays
    sorted_indices = np.argsort(dij)
    dij = dij[sorted_indices]
    xij = xij[sorted_indices]

    hij = IsotonicRegression().fit(dij, xij).predict(dij)

    raw_stress = np.sum(np.square(xij - hij))
    norm_factor = np.sum(np.square(xij))

    kruskal_stress = np.sqrt(raw_stress / norm_factor)
    return kruskal_stress


def evaluate_scaling_kruskal(X_high, X_low, scalars):
    """
    Interface for non-metric stress to match evaluate_scaling function. 
    Note that non-metric stress is scale invariant.
    """
    D_high = pdist(X_high)
    D_low = pdist(X_low)
    return [compute_stress_kruskal(D_high, a * D_low) for a in scalars]


def evaluate_scaling(X_high, X_low, scalars):
    """Function that scales the Stress score within an "interesting" range."""
    D_high = pdist(X_high)
    D_low = pdist(X_low)
    stresses = []
    for scalar in scalars:
        D_low_scaled = D_low * scalar
        result = stress.measure(X_high, X_low, distance_matrices=(
                                D_high, D_low_scaled))
        stresses.append(result['stress'])
    return stresses


def find_min_stress_exact(X, Y):
    """
    Given high dimensional data Y, low dimensional data X, 
    calculates the minimum value of the stress curve exactly.
    """
    D_low = squareform(pdist(X))
    D_high = squareform(pdist(Y))

    alpha = find_optimal_scalars_exact(D_low, D_high)

    return (evaluate_scaling(Y, X, [alpha])[0], alpha)


def find_optimal_scalars_exact(D_low, D_high):
    """
    Given two distance matrices D_low, D_high, computes 
    the optimal scalar value to multiply D_low by so that 
    the normalized stress between them is minimum.
    """

    D_low_triu = D_low[np.triu_indices(D_low.shape[0], k=1)]
    D_high_triu = D_high[np.triu_indices(D_high.shape[0], k=1)]
    return np.sum(D_low_triu * D_high_triu) / np.sum(np.square(D_low_triu))


def find_intersection(D_X, D_A, D_B):
    """
    Calculate the intersection points of two low-dimensional projections.
    """
    numerator = 2 * np.sum(-D_B * D_X + D_A * D_X)
    denominator = np.sum(np.square(D_A) - np.square(D_B))

    alpha = numerator / denominator

    return alpha
