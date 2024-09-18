# Databricks notebook source
# MAGIC %md
# MAGIC ## INSTALL MLCORE SDK

# COMMAND ----------

# DBTITLE 1,Installing MLCore SDK
# MAGIC %pip install sparkmeasure
# MAGIC %pip install google-auth
# MAGIC %pip install google-cloud-storage
# MAGIC %pip install azure-storage-blob
# MAGIC %pip install azure-identity
# MAGIC #%pip install protobuf==3.17.2
# MAGIC #%pip install numpy==1.19.1

# COMMAND ----------

#%pip install numpy==1.19.1

# COMMAND ----------

from sparkmeasure import StageMetrics, TaskMetrics
from pyspark.sql import functions as F

taskmetrics = TaskMetrics(spark)
stagemetrics = StageMetrics(spark)

taskmetrics.begin()
stagemetrics.begin()

# COMMAND ----------

# DBTITLE 1,Load the YAML config
import yaml
import ast
import pickle
import mlflow
from MLCORE_SDK import mlclient
from pyspark.sql import functions as F
import json

try:
    solution_config = (dbutils.widgets.get("solution_config"))
    solution_config = json.loads(solution_config)
    print("Loaded config from dbutils")
except Exception as e:
    print(e)
    with open('../data_config/SolutionConfig_dev.yaml', 'r') as solution_config:
        solution_config = yaml.safe_load(solution_config)  

# COMMAND ----------

# MAGIC %md
# MAGIC ## PERFORM MODEL TRAINING 

# COMMAND ----------

# DBTITLE 1,Imports
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import time
from sklearn.metrics import *
import json
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from prophet import Prophet
import logging
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import time, json
from utils import utils
from sklearn.metrics import *
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------

try :
    env = dbutils.widgets.get("env")
except :
    env = "dev"
print(f"Input environment : {env}")

# COMMAND ----------

# DBTITLE 1,Input from the user
# GENERAL PARAMETERS
try:
    retrain_params = (dbutils.widgets.get("retrain_params"))
    retrain_params = json.loads(retrain_params)
    print("Loaded Retrain Params from job params")
    is_retrain = True
except:
    is_retrain = False
tracking_env = solution_config["general_configs"]["tracking_env"]
try :
    sdk_session_id = dbutils.widgets.get("sdk_session_id")
except :
    sdk_session_id = solution_config["general_configs"]["sdk_session_id"][tracking_env]

if sdk_session_id.lower() == "none":
    sdk_session_id = solution_config["general_configs"]["sdk_session_id"][tracking_env]
 
tracking_url = solution_config["general_configs"].get("tracking_url", None)
tracking_url = f"https://{tracking_url}" if tracking_url else None

# JOB SPECIFIC PARAMETERS
input_table_configs = solution_config["train"]["datalake_configs"]["input_tables"]
output_table_configs = solution_config["train"]["datalake_configs"]['output_tables']
model_configs = solution_config["train"]["model_configs"]
storage_configs = solution_config["train"]["storage_configs"]
feature_columns = solution_config['train']["feature_columns"]
target_columns = solution_config['train']["target_columns"]
test_size = solution_config['train']["test_size"]
model_name=model_configs.get('model_params', {}).get('model_name', '')
print(f"Model Name : {model_name}")
from MLCORE_SDK.helpers.mlc_helper import get_job_id_run_id
job_id, run_id ,task_id= get_job_id_run_id(dbutils)
print(job_id, run_id)

# COMMAND ----------

def get_name_space(table_config):
    data_objects = {}
    for table_name, config in table_config.items() : 
        catalog_name = config.get("catalog_name", None)
        schema = config.get("schema", None)
        table = config.get("table", None)

        if catalog_name and catalog_name.lower() != "none":
            table_path = f"{catalog_name}.{schema}.{table}"
        else :
            table_path = f"{schema}.{table}"

        data_objects[table_name] = table_path
    
    return data_objects

# COMMAND ----------

input_table_paths = get_name_space(input_table_configs)
output_table_paths = get_name_space(output_table_configs)

# COMMAND ----------

ft_data = spark.sql(f"SELECT * FROM {input_table_paths['input_1']}")
gt_data = spark.sql(f"SELECT * FROM {input_table_paths['input_2']}")

# COMMAND ----------

ft_data.display()

# COMMAND ----------

gt_data.display()

# COMMAND ----------

ft_data.count(), gt_data.count()

# COMMAND ----------

input_table_configs["input_1"]["primary_keys"]

# COMMAND ----------

try : 
    date_filters = dbutils.widgets.get("date_filters")
    print(f"Input date filter : {date_filters}")
    date_filters = json.loads(date_filters)
except :
    date_filters = {}

try : 
    hyperparameters = dbutils.widgets.get("hyperparameters")
    print(f"Input hyper parameters : {hyperparameters}")
    hyperparameters = json.loads(hyperparameters)
except :
    hyperparameters = {}

print(f"Data filters used in model train : {date_filters}, hyper parameters : {hyperparameters}")

# COMMAND ----------

if date_filters and date_filters['feature_table_date_filters'] and date_filters['feature_table_date_filters'] != {} :   
    ft_start_date = date_filters.get('feature_table_date_filters', {}).get('start_date',None)
    ft_end_date = date_filters.get('feature_table_date_filters', {}).get('end_date',None)
    if ft_start_date not in ["","0",None] and ft_end_date not in  ["","0",None] : 
        print(f"Filtering the feature data")
        ft_data = ft_data.filter(F.col("timestamp") >= int(ft_start_date)).filter(F.col("timestamp") <= int(ft_end_date))

if date_filters and date_filters['ground_truth_table_date_filters'] and date_filters['ground_truth_table_date_filters'] != {} : 
    gt_start_date = date_filters.get('ground_truth_table_date_filters', {}).get('start_date',None)
    gt_end_date = date_filters.get('ground_truth_table_date_filters', {}).get('end_date',None)
    if gt_start_date not in ["","0",None] and gt_end_date not in ["","0",None] : 
        print(f"Filtering the ground truth data")
        gt_data = gt_data.filter(F.col("timestamp") >= int(gt_start_date)).filter(F.col("timestamp") <= int(gt_end_date))

# COMMAND ----------

ground_truth_data = gt_data.select([input_table_configs["input_2"]["primary_keys"]] + target_columns)
features_data = ft_data.select([input_table_configs["input_1"]["primary_keys"]] + feature_columns + ['Country'])

# COMMAND ----------

# DBTITLE 1,Joining Feature and Ground truth tables on primary key
final_df = features_data.join(ground_truth_data, on = input_table_configs["input_1"]["primary_keys"])

# COMMAND ----------

# DBTITLE 1,Converting the Spark df to Pandas df
final_df_pandas = final_df.toPandas()
final_df_pandas.head()

# COMMAND ----------

final_df_pandas.shape

# COMMAND ----------

# DBTITLE 1,Dropping the null rows in the final df
final_df_pandas.dropna(inplace=True)

# COMMAND ----------

final_df_pandas.shape

# COMMAND ----------

def to_date_(col):
    """
    Checks col row-wise and returns first date format which returns non-null output for the respective column value
    """
    formats = (
        "MM-dd-yyyy",
        "dd-MM-yyyy",
        "MM/dd/yyyy",
        "yyyy-MM-dd",
        "M/d/yyyy",
        "M/dd/yyyy",
        "MM/dd/yy",
        "MM.dd.yyyy",
        "dd.MM.yyyy",
        "yyyy-MM-dd",
        "yyyy-dd-MM",
    )
    return F.coalesce(*[F.to_date(col, f) for f in formats])

# COMMAND ----------

def add_date_timestamp_id_columns(spark_df):
    """Add timestamp, date, and id columns to the Spark DataFrame."""
    now = datetime.now()
    date = now.strftime("%m-%d-%Y")
    spark_df = spark_df.withColumn("timestamp", F.expr("reflect('java.lang.System', 'currentTimeMillis')").cast("long"))
    spark_df = spark_df.withColumn("date", F.lit(date))
    spark_df = spark_df.withColumn("date", to_date_(F.col("date")))
    
    # Add a monotonically increasing column if not present
    if "id" not in spark_df.columns:
        window = Window.orderBy(F.monotonically_increasing_id())
        spark_df = spark_df.withColumn("id", F.row_number().over(window))

    return spark_df

# COMMAND ----------

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {"r2": r2, "mse": mse, "mae": mae, "rmse": rmse}

# COMMAND ----------

# DBTITLE 1,UDF
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from pyspark.sql.types import *
from pyspark.sql.functions import col
import pickle
import base64

# Define the schema for the output DataFrame
result_schema = StructType([
    StructField("Country", StringType(), True),
    StructField("output_df", StringType(), True),  # JSON string of the output DataFrame
    StructField("model", StringType(), True),  # Serialized model string
    StructField("train_metrics", StringType(), True),
    StructField("test_metrics", StringType(), True),
    StructField("example_input", StringType(), True)
])

def train_udf(df):
    

    X = df[feature_columns]  # feature (time in numeric form)
    y = df['selling_price']  # target (price)

    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # Fit the Linear Regression model
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(X_train, y_train)

    example_input = X_train[:5].to_json(orient='records')

    # Make predictions
    y_pred_train = lin_reg_model.predict(X_train).tolist()
    y_pred_test = lin_reg_model.predict(X_test).tolist()

    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_pred_train)
    test_metrics = calculate_metrics(y_test, y_pred_test)

    # Create DataFrames for train and test with predictions
    train_df = X_train.copy()
    test_df = X_test.copy()
    train_df["y"] = y_train
    test_df["y"] = y_test
    train_df["prediction"] = y_pred_train
    test_df["prediction"] = y_pred_test

    train_df["dataset_type_71E4E76EB8C12230B6F51EA2214BD5FE"] = "train"
    test_df["dataset_type_71E4E76EB8C12230B6F51EA2214BD5FE"] = "test"

    # Concatenate the DataFrames vertically
    output_df = pd.concat([train_df, test_df], ignore_index=True)

    # Rename the columns back to original
    output_df = output_df.rename(columns={'ds_numeric': 'epoch', 'y': 'price'})

    # Serialize the model
    serialized_model = base64.b64encode(pickle.dumps(lin_reg_model)).decode('utf-8')
    
    # Convert the DataFrame to a JSON string
    output_json_str = output_df.to_json(orient='records', date_format='iso')
    Country = df["Country"].iloc[0] if 'Country' in df.columns else "Unknown"
    
    # Create a DataFrame with a single row containing the JSON string and the serialized model
    result_df = pd.DataFrame(
        {
            "Country": [Country],
            "output_df": [output_json_str],
            "model": [serialized_model],
            "train_metrics": [json.dumps(train_metrics)],
            "test_metrics": [json.dumps(test_metrics)],
            "example_input": [example_input]
        }
    )
    
    return result_df


# COMMAND ----------

#df = final_df.limit(1000)
output_df = final_df.groupBy("Country").applyInPandas(
    train_udf,
    schema=result_schema
)

# COMMAND ----------

pandas_df = output_df.toPandas()

# COMMAND ----------

pandas_df

# COMMAND ----------

import json
import pickle
import base64

# Initialize the dictionary to store the results
Country_dict = {}

for index, row in pandas_df.iterrows():
    Country = row['Country']
    output_df_list = json.loads(row['output_df'])
    model_bytes = base64.b64decode(row['model'])
    model = pickle.loads(model_bytes)
    
    # Add the data to the dictionary
    Country_dict[Country] = {
        'output_df': output_df_list,
        'model': model,
        'train_metrics': json.loads(row['train_metrics']),
        'test_metrics': json.loads(row['test_metrics']),
        'example_input': json.loads(row['example_input'])
    }


print(Country_dict)


# COMMAND ----------

model_name=model_configs.get('model_params', {}).get('model_name', '')
model_name

# COMMAND ----------

# DBTITLE 1,Displaying the test metrics 
db_name = output_table_configs["output_1"]["schema"]
catalog_name = output_table_configs["output_1"]["catalog_name"]
output_path = output_table_paths["output_1"]

# COMMAND ----------

train_data_date_dict = {
    "feature_table" : {
        "ft_start_date" : ft_data.select(F.min("timestamp")).collect()[0][0],
        "ft_end_date" : ft_data.select(F.max("timestamp")).collect()[0][0]
    },
    "gt_table" : {
        "gt_start_date" : gt_data.select(F.min("timestamp")).collect()[0][0],
        "gt_end_date" : gt_data.select(F.max("timestamp")).collect()[0][0]        
    }
}

# COMMAND ----------

if input_table_configs["input_1"]["catalog_name"]:
    feature_table_path = input_table_paths["input_1"]
else:
    feature_table_path = spark.sql(f"desc formatted {input_table_paths['input_1']}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]

if input_table_configs["input_2"]["catalog_name"]:
    gt_table_path = input_table_paths["input_2"]
else:
    gt_table_path = spark.sql(f"desc formatted {input_table_paths['input_2']}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]


print(feature_table_path, gt_table_path)

# COMMAND ----------

# Add job_run_add method for 1st model
mlclient.log(
    operation_type="job_run_add", 
    session_id = sdk_session_id, 
    dbutils = dbutils, 
    request_type = "train", 
    job_config = 
    {
        "table_name" : output_table_configs["output_1"]["table"],
        "model_name" : f"{model_name}_USA",
        "feature_table_path" : feature_table_path,
        "ground_truth_table_path" : gt_table_path,
        "feature_columns" : feature_columns,
        "target_columns" : target_columns,
        "model" : model_name,
        "model_runtime_env" : "python",
        "reuse_train_session" : False
    },
    tracking_env = env,
    tracking_url = tracking_url,
    spark = spark,
    verbose = True,
    )


# COMMAND ----------

from MLCORE_SDK.sdk.manage_sdk_session import get_session
existing_session_data = get_session(
    sdk_session_id,
    dbutils,
    api_endpoint=tracking_url,
    tracking_env=tracking_env,
).json()["data"]
existing_session_state = existing_session_data.get("state_dict", {})
project_id = existing_session_state.get("project_id", "")
version = existing_session_state.get("version", "")
print(project_id, version)

# COMMAND ----------

from sparkmeasure import StageMetrics, TaskMetrics

# COMMAND ----------

from mlflow.tracking import MlflowClient
def get_latest_model_version(model_configs):
    try : 
        mlflow_uri = model_configs.get("model_registry_params").get("host_url")
        model_name = model_configs.get("model_params").get("model_name")
        mlflow.set_registry_uri(mlflow_uri)
        client = MlflowClient()
        x = client.get_latest_versions(model_name)
        model_version = x[0].version
        return model_version
    except Exception as e :
        print(f"Exception in {get_latest_model_version} : {e}")
        return 0

# COMMAND ----------


for key, value in Country_dict.items():

    #start tastmetrics and stage metrics
    taskmetrics = TaskMetrics(spark)
    stagemetrics = StageMetrics(spark)
    taskmetrics.begin()
    stagemetrics.begin()

    #1. Write the train_output_tables to HIVE
    Country=key
    table_name = f"{model_name}_{Country}_train_output"
    if catalog_name and catalog_name.lower() != "none":
        spark.sql(f"USE CATALOG {catalog_name}")

    # Create the database if it does not exist
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
    print(f"HIVE METASTORE DATABASE NAME : {db_name}")
    
    output_df=value['output_df']
    train_output_df=pd.DataFrame(output_df)
    train_output_df = spark.createDataFrame(train_output_df)
    train_output_df.createOrReplaceTempView(table_name)
    feature_table_exist = [True for table_data in spark.catalog.listTables(db_name) if table_data.name.lower() == table_name.lower() and not table_data.isTemporary]

    output_path = f"{db_name}.{table_name}"
    if not any(feature_table_exist):
        print(f"CREATING SOURCE TABLE")
        spark.sql(f"CREATE TABLE IF NOT EXISTS {output_path} AS SELECT * FROM {table_name}")
    else :
        print(F"UPDATING SOURCE TABLE")
        spark.sql(f"INSERT INTO {output_path} SELECT * FROM {table_name}")

    output_1_table_path = output_path
    

    print(f"Features Hive Path : {output_1_table_path}")



    #end tastmetrics and stage metrics
    stagemetrics.end()
    taskmetrics.end()
    stage_Df = stagemetrics.create_stagemetrics_DF("PerfStageMetrics")
    task_Df = taskmetrics.create_taskmetrics_DF("PerfTaskMetrics")
    compute_metrics = stagemetrics.aggregate_stagemetrics_DF().select("executorCpuTime", "peakExecutionMemory","memoryBytesSpilled","diskBytesSpilled").collect()[0].asDict()
    compute_metrics['executorCpuTime'] = compute_metrics['executorCpuTime']/1000
    compute_metrics['peakExecutionMemory'] = float(compute_metrics['peakExecutionMemory']) /(1024*1024)
    

    #2. Register model in MLCORE
    mlflow.end_run()
    model_artifact_id=mlclient.log(
        operation_type="register_model",
        sdk_session_id=sdk_session_id,
        dbutils=dbutils,
        spark=spark,
        model=model,
        model_name=f"{model_name}_{key}_model_1709_1",
        model_runtime_env="python",
        train_metrics=Country_dict[key]['train_metrics'],
        test_metrics=Country_dict[key]['test_metrics'],
        feature_table_path=feature_table_path,  # Adjust this line if feature_table_path is specific to each Country 
        ground_truth_table_path=gt_table_path,
        train_output_path=output_1_table_path,
        train_output_rows=train_output_df.count(),
        train_output_cols=train_output_df.columns,
        table_schema=train_output_df.schema,
        column_datatype=train_output_df.dtypes,
        feature_columns=feature_columns,
        target_columns=target_columns,
        table_type="unitycatalog" if output_table_configs["output_1"]["catalog_name"] else "internal",
        train_data_date_dict=train_data_date_dict,  
        compute_usage_metrics=compute_metrics,
        taskmetrics=taskmetrics,
        stagemetrics=stagemetrics,
        tracking_env=env,
        # horizon=horizon,
        #frequency=frequency,
        example_input=value['example_input'],
        #signature= Country .get("signature", None),
        # register_in_feature_store=True,  # Uncomment if needed
        model_configs=model_configs,
        tracking_url=tracking_url,
        verbose=True

    )

#     model_version = get_latest_model_version(model_configs) + 1
#     train_output_df = train_output_df.withColumn("model_name", F.lit(model_name).cast("string"))
#     train_output_df = train_output_df.withColumn("model_version", F.lit(model_version).cast("string"))
#     train_output_df = train_output_df.withColumn("train_job_id", F.lit(job_id).cast("string"))
#     train_output_df = train_output_df.withColumn("train_run_id", F.lit(run_id).cast("string"))
#     train_output_df = train_output_df.withColumn("train_task_id",F.lit(task_id).cast("string"))



#     aggregated_table_name = f"{sdk_session_id}_{key}_train_output_aggregated_table"
#     catalog_name = output_table_configs["output_1"]["catalog_name"]
#     if catalog_name and catalog_name.lower() != "none":
#         output_path = f"{catalog_name}.{db_name}.{aggregated_table_name}"
#     else :
#         output_path = f"{db_name}.{aggregated_table_name}"

# # Add Model Artifact ID retrieved after registering the model.
#         train_output_df = train_output_df.withColumn("model_artifact_id", F.lit(model_artifact_id))

# # Get the catalog name from the table name
#     if catalog_name and catalog_name.lower() != "none":
#         spark.sql(f"USE CATALOG {catalog_name}")

# # Create the database if it does not exist
#     spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
#         print(f"HIVE METASTORE DATABASE NAME : {db_name}")

# train_table_exists = [True for table_data in spark.catalog.listTables(db_name) if table_data.name.lower() == aggregated_table_name.lower() and not table_data.isTemporary]

# # IF the table exists, reset the ID based on existing max marker
# if any(train_table_exists):
#     max_id_value = spark.sql(f"SELECT max(id) FROM {output_path}").collect()[0][0]
#     window = Window.orderBy(F.monotonically_increasing_id())
#     train_output_df = train_output_df.withColumn("id", F.row_number().over(window) + max_id_value)

# # Create temporary View.
# train_output_df.createOrReplaceTempView(aggregated_table_name)

# if not any(train_table_exists):
#   print(f"CREATING SOURCE TABLE")
#   spark.sql(f"CREATE TABLE IF NOT EXISTS {output_path} PARTITIONED BY (model_artifact_id) AS SELECT * FROM {aggregated_table_name}")
# else :
#   print(F"UPDATING SOURCE TABLE")
#   spark.sql(f"INSERT INTO {output_path} PARTITION (model_artifact_id) SELECT * FROM {aggregated_table_name}")

# aggregated_table_path = output_path

# print(f"Aggregated Train Output Hive Path : {aggregated_table_path}")


#     # Register Aggregate Train Output in MLCore
#     mlclient.log(operation_type = "register_table",
#     sdk_session_id = sdk_session_id,
#     dbutils = dbutils,
#     spark = spark,
#     table_name = aggregated_table_name,
#     num_rows = train_output_df.count(),
#     tracking_env = tracking_env,
#     cols = train_output_df.columns,
#     column_datatype = train_output_df.dtypes,
#     table_schema = train_output_df.schema,
#     primary_keys = ["id"],
#     table_path = aggregated_table_path,
#     table_type="unitycatalog" if output_table_configs["output_1"]["catalog_name"] else "internal",
#     table_sub_type="Train_Output",
#     tracking_url = tracking_url,
#     platform_table_type = "Aggregated_train_output",
#     verbose=True)

