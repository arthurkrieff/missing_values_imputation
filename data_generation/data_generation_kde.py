import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KernelDensity
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

def compute_kde(serie):
    """
    Function that creates the Kernel Density Estimate from a pd.Series
    Input: 
        - serie (pd.Serie): ordered serie per time
    Output:
        - KernelDensity object
    """
    values = serie.replace([np.inf, -np.inf], np.nan). \
                    dropna(). \
                    values. \
                    reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.001).fit(values)
    return kde

def sample_from_kde(kde, n=1):
    """
    Function that samples a value based on a KernelDensity Object
    Intput:
        - kde (KernelDensity object)
    Output:
        - float 
    """
    value = kde.sample(n)
    return value[0][0]

def generate_prices_from_kde(kde, start_price, n_samples=2851):
    """
    Function that generates a new time serie based on KernelDensity
    Input:
        - kde: KernelDensity Object 
        - start_price: float
        - n_samples: integer, represents the time serie length
    """
    prices = []
    for i in range(n_samples):
        try:
            ret = sample_from_kde(kde, n=1)
            price = price * (1+ret)
            prices.append(price)
        except NameError:
            price = start_price
            prices.append(price)
    return prices

def get_returns(serie):
    """
    Function that computes the returns of a pandas serie
    Input : 
        - serie (pd.Series) of values
    Output:
        - serie (pd.Series) with returns (evolution of values)
    """
    return_serie = (serie-serie.shift(1)) / serie.shift(1)
    return return_serie


if __name__ == "__main__":
    # Load Data
    df_challenge = import_datasets()[0]
    df_challenge.set_index("Date", inplace=True)
    
    all_data = dict()
    
    # For each asset
    for col in tqdm(df_challenge.columns):
        # Get values
        serie = df_challenge.loc[:, col]
        start_price = serie.dropna().values[0]

        # Compute returns
        returns = get_returns(serie)

        # Kernel Estimation
        kde = compute_kde(returns)

        # Generate prices
        data = generate_prices_from_kde(kde, start_price, n_samples=2851)
        all_data[col] = pd.Series(data)
        
    full = pd.concat(all_data.values(), axis=1)
    full_spark = spark.createDataFrame(full)
    
    # Write on Workspace
    workspace_parquet_file_url = f'wasbs://{workspace_container}@hecdf.blob.core.windows.net/{myname}/generated_data_kde.parquet'
    print('Writing of the file in the workspace in parquet')
    full_spark.write.mode('overwrite').parquet(workspace_parquet_file_url)
    print("Success")
    