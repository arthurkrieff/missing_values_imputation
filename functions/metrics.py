import numpy as np
import pandas as pd
from functions.utils import find_first_value

def nrmse_adjusted(X_true, X_imputed, X_not_imputed):
    """
    Function that computes the NRMSEs from a Matrix. It is adjusted because it
    doesn't take into account the historical NaNs.
    Input:
        - X_true: np.array with the true values
        - X_imputed: np.array with the imputed values
        - X_not_imputed : np.array with missing values 
    Output:
        - nrmses: Python Dictionary where key = index of the column and
        val = tuple (x1, x2) where x1 is the nrmse and x2 is the count of imputed values
    """
    nrmses = dict()

    for col in range(X_true.shape[1]):
        squared_error = 0
        count_values = 0
        first_val = find_first_value(X_not_imputed[:, col])
        if first_val == "NaN":
            continue
        else:
            for row, _ in np.ndenumerate(X_not_imputed[first_val:, col]):
                idx = first_val + row[0]
                if np.isnan(X_not_imputed[idx, col]):
                    squared_error += (X_true[idx, col] - X_imputed[idx, col])**2
                    count_values += 1
            if count_values != 0:
                variance = np.nanvar(X_true[:, col])
                nrmse = np.sqrt((squared_error/count_values) / variance)
                nrmses[col] = (nrmse, count_values)

    return nrmses


def compute_cov_matrix(X):
    """
    Function that compute the covariance matrix of an np.array
    Input:
        - X : np.array, with NaN or not
    Output:
        - Covariance matrix: np.array
    """
    # We convert to pandas because pandas deal well with missing values in cov matrix
    df_cov = pd.DataFrame(X).cov()
    df_cov = df_cov.fillna(0)
    return df_cov.values


def frobenius_norm_cov_matrices(X_true, X_imputed):
    """
    Computes the frobenius norm of the difference between X_true and X_imputed
    Input:
        - X_true: np.array
        - X_imputed: np.array
    Output:
        - float : frobenius norm value
    """
    # Compute Cov Matrix
    corr_true = compute_cov_matrix(X_true)
    corr_imputed = compute_cov_matrix(X_imputed)
    # Compute norm
    norm = np.linalg.norm(corr_true - corr_imputed)
    return norm