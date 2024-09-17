# Databricks notebook source
# MAGIC %pip install /dbfs/FileStore/Bayada/package/MLCoreSDK-0.4.5-py3-none-any.whl --force-reinstall

# COMMAND ----------

# MAGIC %pip install deepchecks
# MAGIC # %pip install numpy==1.19.1
# MAGIC # %pip install pandas==1.0.5
# MAGIC # %pip install matplotlib==3.3.2

# COMMAND ----------

# MAGIC %pip install azure-storage-blob

# COMMAND ----------

# MAGIC %pip install azure-identity

# COMMAND ----------

table_path = dbutils.widgets.get("table_path") #"super_store_hive_db_2.raw_super_store_feature_2"

# COMMAND ----------

dataset = spark.sql(f"SELECT * FROM {table_path}")

# COMMAND ----------

# Splitting the data to train/test set
trainDF, testDF = dataset.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

import yaml
import json
import numpy as np

# COMMAND ----------

def detect_categorical_cols(df, threshold=5):
    """
    Get the Categorical columns with greater than threshold percentage of unique values.

    This function returns the Categorical columns with the unique values in the column
    greater than the threshold percentage.

    Parameters
    ----------
    df: pyspark.sql.DataFrame
    threshold : int , default = 5
        threshold value in percentage

    Returns
    -------
    report_data : dict
        dictionary containing the Numeric column data.

    """
    df = df.toPandas()
    no_of_rows = df.shape[0]
    possible_cat_cols = (
        df.convert_dtypes()
        .select_dtypes(exclude=[np.datetime64, "float", "float64"])
        .columns.values.tolist()
    )
    temp_series = df[possible_cat_cols].apply(
        lambda col: (len(col.unique()) / no_of_rows) * 100 > threshold
    )
    cat_cols = temp_series[temp_series == False].index.tolist()
    return cat_cols

# COMMAND ----------

cat_cols = detect_categorical_cols(dataset)

# COMMAND ----------

from deepchecks.tabular import Dataset
import time
pd_train = trainDF.toPandas()
pd_test = testDF.toPandas()

ds_train = Dataset(pd_train, cat_features=cat_cols)
ds_test = Dataset(pd_test, cat_features=cat_cols)

# COMMAND ----------

from deepchecks.tabular.suites import data_integrity
# Validate the training set
train_res = data_integrity().run(ds_train)

# COMMAND ----------

report_path = f"/dbfs/FileStore/Amplify/Data_Integrity_Report_{int(time.time())}.html"
train_res.save_as_html(report_path)

# COMMAND ----------

storage_configs = {
    "cloud_provider" : "azure",
    "params" :
        {
          "storage_account_name" : "mlcdevtigerstorage38173",
          "container_name" : "mlcore" 
        }
}


# COMMAND ----------

storage_configs = json.loads(dbutils.widgets.get("storage_configs"))

# COMMAND ----------

for key, val in storage_configs["params"].items():
    storage_configs[key] = val

del storage_configs["params"]

# COMMAND ----------

job_id = dbutils.widgets.get("job_id")
run_id = dbutils.widgets.get("run_id")
env = dbutils.widgets.get("env")

project_id = "1e5014141c084b028822407e816269bd"
version = "Solution_configs_upgrades"

# COMMAND ----------

from MLCORE_SDK import mlclient
if storage_configs["cloud_provider"] == "databricks_uc":
    catalog_name = storage_configs.get("catalog_name")
    schema_name = storage_configs.get("schema_name")
    volume_name = storage_configs.get("volume_name")

    artifact_path_uc_volume = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/{env}/media_artifacts/{project_id}/{version}/{job_id}/{run_id}/custom_reports/Data_Integrity_Report_{int(time.time())}.html"
    print(artifact_path_uc_volume)
    mlclient.log(
        operation_type = "upload_blob_to_cloud",
        blob_path=report_path,
        dbutils = dbutils ,
        target_path = f"{report_directory}/custom_reports/EdaReport_{int(time.time())}.html",
        resource_type = "databricks_uc",
        project_id = project_id,
        version = version,
        job_id = job_id,
        run_id = run_id,
        request_type = "eda",
        storage_configs = storage_configs,
        tracking_env = env)
else :
    container_name = storage_configs.get("container_name")
    report_directory = f"{env}/media_artifacts/{project_id}/{version}/{job_id}/{run_id}"
    print(report_directory)
    mlclient.log(
        operation_type = "upload_blob_to_cloud",
        source_path=report_path,
        dbutils = dbutils ,
        target_path = f"{report_directory}/custom_reports/EdaReport_{int(time.time())}.html",
        resource_type = "az",
        project_id = project_id,
        version = version,
        job_id = job_id,
        run_id = run_id,
        request_type = "eda",
        storage_configs = storage_configs,
        tracking_env = env)
