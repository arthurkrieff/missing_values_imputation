import pandas as pd
import numpy as np
from numpy import linalg as la
from functions.import_data import import_datasets
from blob_credentials import facts_sas_token, facts_container, workspace_sas_token, workspace_container
from pyspark.sql import SparkSession

myname = "marc-samvath-philippe.vigneron"

spark = SparkSession \
    .builder \
    .appName(f"Test-{myname}") \
    .config("spark.executor.instance", "1") \
    .config("spark.executor.memory","512m") \
    .config('spark.jars.packages',"org.apache.hadoop:hadoop-azure:3.1.1") \
    .config("fs.azure", "org.apache.hadoop.fs.azure.NativeAzureFileSystem") \
    .config("fs.wasbs.impl","org.apache.hadoop.fs.azure.NativeAzureFileSystem") \
    .config(f"fs.azure.sas.{facts_container}.hecdf.blob.core.windows.net", facts_sas_token) \
    .config(f"fs.azure.sas.{workspace_container}.hecdf.blob.core.windows.net", workspace_sas_token) \
    .getOrCreate()


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    E = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += E * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False


def MultiBrownMo():
    '''Generate the realization of multivariate Brownian Motion given:
        - time horizon
        - discretization point
        - drift
        - covariance matrix
        - start values
        of the original dataset_challenge of Natixis'''

    # Fixing the random seed for documentation purposes
    np.random.seed(42)

    # Get the datasets
    dataset_challenge = import_datasets()[0]

    # Defining the parameters for the Brownian Motion
    new_cov = nearestPD(dataset_challenge.cov())
    time = dataset_challenge.shape[0]
    assets_num = dataset_challenge.shape[1]-1
    drift = np.mean(dataset_challenge.iloc[:, 1:].diff()).to_numpy()
    start_vals = [dataset_challenge[col].loc[~dataset_challenge[col].isnull()].iloc[0]
                  for col in dataset_challenge.columns if col != 'Date']

    # Initialize an array with Zeros
    brownMo = np.zeros((1, assets_num))

    # Get the Cholesky Decomposition for the formula
    sigma = la.cholesky(new_cov)

    # Update the array with the real initial values
    brownMo[0, :] = np.asarray(start_vals)

    # Fill the other timesteps
    for t in range(1, time):
        new_row = brownMo[t-1, :] + drift \
                + np.random.normal(0, 1, size=(1, assets_num)) @ sigma

        brownMo = np.vstack([brownMo, new_row])

    return brownMo


if __name__ == "__main__":
    # Load and perform a simulation
    dataset_challenge_gbm = MultiBrownMo()
    print('Dataset successfully imported!')
    print('Sucessful realization of Multivariate Brownian Motion Path!')

    # Convert to Spark to save in Workspace
    spark_bm = spark.createDataFrame(pd.DataFrame(dataset_challenge_gbm))

    # Write on Workspace
    workspace_parquet_file_url = f'wasbs://{workspace_container}@hecdf.blob.core.windows.net/{myname}/generated_data_gbm.parquet'
    print('Writing of the Full GBM DataSet file in the workspace in parquet')
    spark_bm.write.mode('overwrite').parquet(workspace_parquet_file_url)
    print("Success")