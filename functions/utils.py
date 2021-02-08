import numpy as np
import pandas as pd

def find_first_value(array):
    """
    Function that returns the index of the first non null value of an np.array
    Input:
        - np.array of 1D
    Output:
        - Integer
    """
    for i, val in enumerate(array):
        if not np.isnan(val):
            return i
    return "NaN"


def to_NAN(df, df_challenge):
    """
    Function that put NaNs values at the same position as df_challenge
    Input:
        - df: pd.DataFrame with no missing values
        - df_challenge: pd.DataFrame with missing values
    Output:
        - pd.DataFrame with missing values
    """
    
    mask = df_challenge.notna()
    return df[mask]


def put_historical_nans(X_imputed, X_not_imputed):
    """
    Function that puts the same historical NaNs as X_not_imputed
    Input:
        - X_imputed: pandas.DataFrame
        - X_not_imputed: pandas.DataFrame
    Output:
        - pandas.DataFrame imputed BUT with historical NaNs
    """
    for col in range(len(X_imputed.columns)):
        first_val = find_first_value(X_not_imputed.iloc[:, col].values)
        if first_val == 0:
            continue
        elif first_val == 'NaN':
            continue
        else:
            X_imputed.iloc[:(first_val-1), col] = np.NaN # Minus 1 because of pandas .loc settings
    return X_imputed