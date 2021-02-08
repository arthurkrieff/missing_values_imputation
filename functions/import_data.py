from pyspark.sql import SparkSession
from azure.storage.blob import ContainerClient
from blob_credentials import facts_sas_token, facts_container, workspace_sas_token, workspace_container
import pandas as pd


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


account_url = "https://hecdf.blob.core.windows.net"

facts_blob_service = ContainerClient(account_url=account_url,
                                     container_name=facts_container,
                                     credential=facts_sas_token)
workspace_blob_service = ContainerClient(account_url=account_url,
                                         container_name=workspace_container,
                                         credential=workspace_sas_token)


def import_datasets(path1='Risques 2/data_set_challenge.csv',
                    path2='Risques 2/final_mapping_candidat.csv',
                    path3='Risques/dataset_final_scenario_4.csv'):
    ''' Imports and returns the data in the following order:
        - dataset_challenge
        - final_mapping_candidat
        - dataset_final_scenario_4 '''

    dataset_challenge = spark.read.csv(f'wasbs://{facts_container}@hecdf.blob.core.windows.net/{path1}',
                                       inferSchema=True,
                                       header=True).toPandas()

    final_mapping_candidat = spark.read.csv(f'wasbs://{facts_container}@hecdf.blob.core.windows.net/{path2}',
                                            inferSchema="true",
                                            header="true").toPandas()

    dataset_final_scenario_4 = spark.read.csv(f'wasbs://{facts_container}@hecdf.blob.core.windows.net/{path3}',
                                              inferSchema="true",
                                              header="true").toPandas()

    dataset_challenge['Date'] = pd.to_datetime(dataset_challenge['Date'])
    dataset_final_scenario_4['Date'] = pd.to_datetime(dataset_final_scenario_4['Date'])

    return [dataset_challenge, final_mapping_candidat, dataset_final_scenario_4]