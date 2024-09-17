# Databricks notebook source
# %pip install /dbfs/FileStore/sdk/dev/MLCoreSDK-0.4.5-py3-none-any.whl --force-reinstall
%pip install sparkmeasure
# %pip install numpy==1.19.1
# %pip install pandas==1.0.5

# COMMAND ----------

from sparkmeasure import StageMetrics, TaskMetrics

taskmetrics = TaskMetrics(spark)
stagemetrics = StageMetrics(spark)

taskmetrics.begin()
stagemetrics.begin()

# COMMAND ----------

try :
    env = dbutils.widgets.get("env")
except :
    env = "dev"
print(f"Input environment : {env}")

# COMMAND ----------

# MAGIC %md <b> User Inputs

# COMMAND ----------

# DBTITLE 1,Load the YAML config
import yaml
import ast
import pickle
from MLCORE_SDK import mlclient
from pyspark.sql import functions as F
import json

try:
    solution_config = (dbutils.widgets.get("solution_config"))
    solution_config = json.loads(solution_config)
    print("Loaded config from dbutils")

except Exception as e:
    print(e)
    with open('../data_config/SolutionConfig.yaml', 'r') as solution_config:
        solution_config = yaml.safe_load(solution_config)  

# COMMAND ----------

# DBTITLE 1,GENERAL PARAMETERS
try :
    sdk_session_id = dbutils.widgets.get("sdk_session_id")
except :
    sdk_session_id = solution_config["general_configs"]["sdk_session_id"][env]

if sdk_session_id.lower() == "none":
    sdk_session_id = solution_config["general_configs"]["sdk_session_id"][env]

# DE SPECIFIC PARAMETERS
batch_size = int(solution_config["data_engineering_ft"].get("batch_size",500))
input_table_configs = solution_config["data_engineering_ft"]["datalake_configs"]["input_tables"]
output_table_configs = solution_config["data_engineering_ft"]["datalake_configs"]['output_tables']

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

# DBTITLE 1,Read input tables
source_1_df = spark.sql(f"SELECT * FROM {input_table_paths['source_1']}")

# COMMAND ----------

# DBTITLE 1,register job in mlcore
mlclient.log(
    operation_type="job_run_add", 
    session_id = sdk_session_id, 
    dbutils = dbutils, 
    request_type = "de", 
    job_config = 
    {
        "table_name" : output_table_configs["output_1"]["table"],
        "table_type" : "Source",
        "batch_size" : batch_size
    },
    tracking_env = env,
    verbose = True,
    spark = spark
    )

# COMMAND ----------

output_1_df = source_1_df.drop('date','id','timestamp')

# COMMAND ----------

output_1_df.display()

# COMMAND ----------

from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def to_date_(col):
    """
    Checks col row-wise and returns first date format which returns non-null output for the respective column value
    """
    formats=(
             "MM-dd-yyyy", "dd-MM-yyyy",
             "MM/dd/yyyy", "yyyy-MM-dd", 
             "M/d/yyyy", "M/dd/yyyy",
             "MM/dd/yy", "MM.dd.yyyy",
             "dd.MM.yyyy", "yyyy-MM-dd",
             "yyyy-dd-MM"
            )
    return F.coalesce(*[F.to_date(col, f) for f in formats])

# COMMAND ----------

# DBTITLE 1,Adding Timestamp and Date Features to a Source 1
now = datetime.now()
date = now.strftime("%m-%d-%Y")
output_1_df = output_1_df.withColumn(
    "timestamp",
    F.expr("reflect('java.lang.System', 'currentTimeMillis')").cast("long"),
)
output_1_df = output_1_df.withColumn("date", F.lit(date))
output_1_df = output_1_df.withColumn("date", to_date_(F.col("date")))


# COMMAND ----------

# DBTITLE 1,writing to output_1
db_name = output_table_configs["output_1"]["schema"]
table_name = output_table_configs["output_1"]["table"]
catalog_name = output_table_configs["output_1"]["catalog_name"]
output_path = output_table_paths["output_1"]

# Get the catalog name from the table name
if catalog_name and catalog_name.lower() != "none": 
  spark.sql(f"USE CATALOG {catalog_name}")
else:
  spark.sql(f"USE CATALOG hive_metastore")

# Create the database if it does not exist
spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
print(f"HIVE METASTORE DATABASE NAME : {db_name}")

feature_table_exist = [True for table_data in spark.catalog.listTables(db_name) if table_data.name.lower() == table_name.lower() and not table_data.isTemporary]

# Fetch the max ID from the existing table if it exists
if any(feature_table_exist):
    max_id_query = f"SELECT MAX(id) as max_id FROM {output_path}"
    max_id = spark.sql(max_id_query).collect()[0]["max_id"]
    print("max_id: ",max_id)
    if max_id is None:
        max_id = 0
else:
    max_id = 0

# Add a monotonically increasing column starting from the previous max ID
if "id" not in output_1_df.columns:
    window = Window.orderBy(F.monotonically_increasing_id())
    output_1_df = output_1_df.withColumn("id", F.row_number().over(window) + max_id)

# Register the DataFrame as a temporary view
output_1_df.createOrReplaceTempView(table_name)

if not any(feature_table_exist):
  print(f"CREATING SOURCE TABLE")
  spark.sql(f"CREATE TABLE IF NOT EXISTS {output_path} AS SELECT * FROM {table_name}")
else :
  print(F"UPDATING SOURCE TABLE")
  spark.sql(f"INSERT INTO {output_path} SELECT * FROM {table_name}")

output_1_table_path = output_path

print(f"Features Hive Path : {output_1_table_path}")

# COMMAND ----------

stagemetrics.end()
taskmetrics.end()

stage_Df = stagemetrics.create_stagemetrics_DF("PerfStageMetrics")
task_Df = taskmetrics.create_taskmetrics_DF("PerfTaskMetrics")

compute_metrics = stagemetrics.aggregate_stagemetrics_DF().select("executorCpuTime", "peakExecutionMemory","memoryBytesSpilled","diskBytesSpilled").collect()[0].asDict()

compute_metrics['executorCpuTime'] = compute_metrics['executorCpuTime']/1000
compute_metrics['peakExecutionMemory'] = float(compute_metrics['peakExecutionMemory']) /(1024*1024)

# COMMAND ----------

# MAGIC %md <b> Use MLCore SDK to register Features and Ground Truth Tables

# COMMAND ----------

table_description = "This is the source table for a marketing campaign use case designed for a retail store, containing raw features such as purchases and customer details. It will be used for feature engineering to build a classification model in the marketing and customer segmentation domain tailored specifically for a retail environment. The table comprises 2240 records and 20 columns."

# COMMAND ----------

# DBTITLE 1,register output 1 in mlcore
mlclient.log(operation_type = "register_table",
    sdk_session_id = sdk_session_id,
    dbutils = dbutils,
    spark = spark,
    table_name = output_table_configs["output_1"]["table"],
    num_rows = output_1_df.count(),
    cols = output_1_df.columns,
    column_datatype = output_1_df.dtypes,
    table_schema = output_1_df.schema,
    primary_keys = output_table_configs["output_1"]["primary_keys"],
    table_path = output_1_table_path,
    table_type="unitycatalog" if output_table_configs["output_1"]["catalog_name"] else "internal" ,
    table_sub_type="Source",
    tracking_env = env,
    compute_usage_metrics = compute_metrics,
    table_description = table_description,
    taskmetrics=taskmetrics,
    stagemetrics=stagemetrics,
    # register_in_feature_store=True,
    verbose=True,)

# COMMAND ----------

storage_configs = solution_config["data_engineering_ft"]["storage_configs"]

# COMMAND ----------

import json

# COMMAND ----------

def get_job_id_run_id(dbutils):
    try:
        notebook_info = json.loads(
            dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson()
        )

        multitaskParentRunId = notebook_info["tags"].get("multitaskParentRunId", None)
        idInJob = notebook_info["tags"].get("idInJob", None)
        print(f"multitaskParentRunId : {multitaskParentRunId}, idInJob: {idInJob}")
        run_id = multitaskParentRunId if multitaskParentRunId else idInJob
        job_id = notebook_info["tags"]["jobId"]
    except:
        job_id = None
        run_id = None

    return job_id, run_id


# COMMAND ----------

job_id, run_id = get_job_id_run_id(dbutils)

# COMMAND ----------

try:
    dbutils.notebook.run(
        "DataAnalysis", 
        timeout_seconds=0,
        arguments={
            "table_path" : input_table_paths['source_1'],
            "storage_configs" : json.dumps(storage_configs),
            "env" : env,
            "job_id": job_id,
            "run_id" : run_id
        })
except:
    pass
