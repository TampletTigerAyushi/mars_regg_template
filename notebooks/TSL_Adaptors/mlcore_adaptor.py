import json
from datetime import datetime

import pandas as pd
from tenacity import retry, stop_after_attempt, stop_after_delay, wait_exponential
from delta.tables import *
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# DONT SHOW SETTINGWITHCOPY WARNING
pd.options.mode.chained_assignment = None
from monitoring.utils import utils
from monitoring.utils import uc_utils
from monitoring.utils.secret_mapping import SECRET_MAPPING
from databricks import sql
from TSL_Adaptors.table_helpers import *


# Adaptor logic : transform the input table into MLCore standard DFs
def read_data_mlcore(dbutils, spark,catalog_details,db_name, table_details={}):

    table_path = table_details.get("table_path", None)
    feature_columns = table_details.get("feature_columns", None)
    gt_column = table_details.get("gt_column", None)
    prediction_column = table_details.get("prediction_column", None)
    primary_keys = table_details.get("primary_keys", None)
    partition_keys = table_details.get("partition_keys", None)

    source_df_sql = get_read_data_delta_sql(data_path=table_path["source_table_path"],table_id=table_path['source_table_id'], spark=spark, dbutils=dbutils)
    if source_df_sql is not None:
        source_df_table = source_df_sql.split("FROM")[1].strip()
        source_df_sql =  source_df_sql + " LIMIT 5000"
    source_df = read_data(
        data_path=table_path["source_table_path"],table_id=table_path['source_table_id'], spark=spark, dbutils=dbutils,sql_query_string=source_df_sql
    )
    gt_df_sql = get_read_data_delta_sql(data_path=table_path["ground_truth_table_path"], table_id=table_path['ground_truth_table_id'],spark=spark, dbutils=dbutils)
    if gt_df_sql is not None:
        gt_df_table = gt_df_sql.split("FROM")[1].strip()
        gt_df_sql = gt_df_sql + " LIMIT 5000"
    gt_df = read_data(
        data_path=table_path["ground_truth_table_path"], table_id=table_path['ground_truth_table_id'],spark=spark, dbutils=dbutils, sql_query_string=gt_df_sql
    )
    inference_df_sql = get_read_data_delta_sql( data_path=table_path["inference_table_path"], table_id=table_path['inference_table_id'],spark=spark, dbutils=dbutils)
    if inference_df_sql is not None:
        inference_df_table = inference_df_sql.split("FROM")[1].strip()
        inference_df_sql = inference_df_sql + " LIMIT 5000"
    inference_df = read_data(
        data_path=table_path["inference_table_path"], table_id=table_path['inference_table_id'],spark=spark, dbutils=dbutils,sql_query_string=inference_df_sql
    )
    if table_path["inference_task_log_table_id"] is not None:
        table_id = table_path.get("inference_task_log_table_id",None)
    else:
        table_id =  None
    inference_task_log_df = read_data(
        data_path=table_path["inference_task_log_path"],
        table_id = table_id,
        spark=spark,
        dbutils=dbutils,
        is_platform_table=True,
    )

    hexdigest = "71E4E76EB8C12230B6F51EA2214BD5FE"
    column_name = f"dataset_type_{hexdigest}"

    if column_name in source_df.columns:
        source_df = source_df.filter(F.col(column_name) == "train")
    elif "datatype" in source_df.columns:
        source_df = source_df.filter(F.col("datatype") == "train")

    if len(partition_keys) > 0:
        source_df = source_df.filter(
            F.col("partition_name_71E4E76EB8C12230B6F51EA2214BD5FE")
            == partition_keys[0]
        )

    reference_df = source_df.join(
        inference_df.select(*primary_keys, *prediction_column),
        on=primary_keys,
        how="inner",
    )

    reference_df = reference_df.join(
        gt_df.select(*primary_keys, gt_column[0]).withColumnRenamed(
            gt_column[0], "y_actual"
        ),
        on=primary_keys,
        how="inner",
    )
    
    #Reference_DF join query
    if (source_df_sql is not None) and (inference_df_sql is not None) and (gt_df_sql is not None):
        join_condition = " AND ".join([f"source.{pk} = inference.{pk}" for pk in primary_keys])
        final_join_condition = " AND ".join([f"source.{pk} = gt.{pk}" for pk in primary_keys])
        prediction_columns_selection = ", ".join([f"inference.{col}" for col in prediction_column])
        reference_df_sql = f"""
            SELECT 
                DISTINCT source.*, 
                {prediction_columns_selection}, 
                gt.{gt_column[0]} AS y_actual 
            FROM 
                {source_df_table} AS source
            INNER JOIN 
                {inference_df_table} AS inference 
            ON 
                {join_condition}
            INNER JOIN 
                {gt_df_table} AS gt
            ON 
                {final_join_condition}
            ORDER BY
                source.id
            LIMIT 5000;
            """
        print("REFERENCE BATCH SQL",reference_df_sql)
        reference_df = read_data(data_path=table_path["inference_table_path"], table_id=table_path['inference_table_id'],spark=spark, dbutils=dbutils,sql_query_string=reference_df_sql)
    
    current_batch_df, required_entry = get_inference_batch(
        dbutils,
        spark,
        table_path["monitoring_task_log_path"],
        inference_task_log_df,
        inference_df,
        catalog_details,
        db_name,
        start_date="",
        end_date="",
        inference_df_sql=inference_df_sql,
        inference_df_table=inference_df_table,
        gt_df_sql=gt_df_sql,
        gt_df_table=gt_df_table,
        primary_keys=primary_keys,
        prediction_column=prediction_column,
        gt_column=gt_column,
        table_path=table_path,
    )

    if "y_actual" not in current_batch_df.columns:
        current_batch_df = current_batch_df.join(
            gt_df.select(*primary_keys, gt_column[0]),
            on=primary_keys,
            how="left",
        ).withColumnRenamed(gt_column[0], "y_actual")

    return reference_df, current_batch_df, required_entry


def get_inference_batch(
    dbutils, spark, task_log_path, df_inf_task, inference_df,catalog_details,db_name, start_date="", end_date="",inference_df_sql=None,inference_df_table=None,gt_df_sql=None,gt_df_table=None,primary_keys=None,prediction_column=None,gt_column=None,table_path=None
):
    """
    Gets the appropriate batch for monitoring subtype
    """

    model_inference_info = dbutils.widgets.get("model_inference_info")
    if isinstance(model_inference_info, str):
        model_inference_info = json.loads(model_inference_info)

    # Fetching task log table for the monitoring subtype
    df_task = get_monitor_task_log_table(dbutils, spark, task_log_path,catalog_details,db_name)
    # Determining max marker
    if df_task == None:
        max_marker = 0
    elif model_inference_info.get("inference_task_log_table_id", None):
        max_marker = int(df_task.select(F.max("inference_id")).collect()[0][0])
    else:
        if df_task.select(F.max("end_marker")).collect()[0][0]:
            max_marker = int(df_task.select(F.max("end_marker")).collect()[0][0])
        else:
            max_marker = 0
    # Applying date filters if configured by the user
    # start_date and end_date are provided in microseconds from API but in the task log, we have timestamps in milliseconds
    # Hence, while applying filter, we divide epoch by 1000 to convert microseconds to milliseconds
    # if inference task log tabel is provided by user, then get the next batch details from id
    marker_col = (
        "id"
        if model_inference_info.get("inference_task_log_table_id", None)
        else "start_marker"
    )
    if start_date not in ["", None] and end_date not in ["", None]:
        df_inf_task = (
            df_inf_task.filter(F.col(marker_col) > max_marker)
            .filter(F.col("timestamp") >= int(start_date) / 1000)
            .filter(F.col("timestamp") <= int(end_date) / 1000)
            .orderBy("date", "timestamp", ascending=True)
        )

    elif start_date not in ["", None]:
        df_inf_task = (
            df_inf_task.filter(F.col(marker_col) > max_marker)
            .filter(F.col("timestamp") >= int(start_date) / 1000)
            .orderBy("date", "timestamp", ascending=True)
        )

    elif end_date not in ["", None]:
        df_inf_task = (
            df_inf_task.filter(F.col(marker_col) > max_marker)
            .filter(F.col("timestamp") <= int(end_date) / 1000)
            .orderBy("date", "timestamp", ascending=True)
        )

    else:
        df_inf_task = df_inf_task.filter(F.col(marker_col) > max_marker).orderBy(
            "id", ascending=True
        )
    # Loading the data as per the Inference batch
    if df_inf_task.first() != None:
        required_entry = df_inf_task.first()
        start_marker = extract_integer(required_entry["start_marker"])
        end_marker = extract_integer(required_entry["end_marker"])

        inference_current_batch = (
            inference_df.filter(F.col("id") >= start_marker)
            .filter(F.col("id") <= end_marker)
            .orderBy("id")
        )
        
        #Current_DF SQL JOIN Query
        if (inference_df_sql is not None) and (gt_df_sql is not None):
            final_join_condition = " AND ".join([f"inference.{pk} = gt.{pk}" for pk in primary_keys])
            prediction_columns_selection = ", ".join([f"inference.{col}" for col in prediction_column])
            current_df_sql = f"""
            SELECT 
                DISTINCT inference.*, 
                gt.{gt_column[0]} AS y_actual
            FROM 
                {inference_df_table} AS inference 
            INNER JOIN 
                {gt_df_table} AS gt
            ON 
                {final_join_condition}
            WHERE
                inference.id >= {start_marker}
                AND 
                inference.id <= {end_marker}
            ORDER BY
                inference.id
            """
            print("CURRENT BATCH SQL",current_df_sql)
            inference_current_batch = read_data(data_path=table_path["inference_table_path"], table_id=table_path['inference_table_id'],spark=spark, dbutils=dbutils,sql_query_string=current_df_sql)


        # Dropping vector features as it is not of supported datatypes in Pandas
        if "vector_features" in inference_current_batch.columns:
            inference_current_batch = inference_current_batch.drop("vector_features")
        if "rawPrediction" in inference_current_batch.columns:
            inference_current_batch = inference_current_batch.drop("rawPrediction")
        return inference_current_batch, required_entry
    else:
        raise Exception("Inference batches are exhausted, Skipping remaining cells")


def get_monitor_task_log_table(dbutils, spark, task_log_path,catalog_details,db_name):
    """
    Fetches the task log table for the input monitoring subtype
    """
    model_inference_info = dbutils.widgets.get("model_inference_info")
    if isinstance(model_inference_info, str):
        model_inference_info = json.loads(model_inference_info)
    run_id, job_id, table, source = fetch_id(dbutils)
    project_id, version = (model_inference_info["project_id"],model_inference_info["version"],)
    table_exists_response = table_exists("Task_Log","Monitor_Batch",job_id,version,project_id,dbutils)
    if table_exists_response:
        print(f"Task log table exists")
        table_id = table_exists_response["data"][0]["table_id"]
        print("MONITORING TASK LOG TABLE ID", table_id)
        df_task = read_data(
            data_path=task_log_path,
            spark=spark,
            dbutils=dbutils,
            table_id=table_id,
            is_platform_table=True,
        )
        return df_task
    else:
        print("Task log table does not exist, Creating the task log table")
        max_id = 1
        inference_id = 1
        # deployment env
        now = datetime.now()
        date = now.strftime("%m-%d-%Y")
        run_id, job_id, table, source = fetch_id(dbutils)
        schema = StructType(
            [
                StructField("monitoring_subtype", StringType(), True),
                StructField("start_marker", IntegerType(), True),
                StructField("end_marker", IntegerType(), True),
                StructField("table_name", StringType(), True),
            ]
        )

        df_record = [("-1", None, None, None)]
        df_task = spark.createDataFrame(df_record, schema=schema)
        df_task = df_task.withColumn(
            "timestamp",
            F.expr("reflect('java.lang.System', 'currentTimeMillis')").cast("long"),
        )
        df_task = df_task.withColumn("date", F.lit(date))
        df_task = df_task.withColumn("date", F.to_date(F.col("date"), "MM-dd-yyyy"))
        df_task = df_task.withColumn("job_id", F.lit(job_id))
        df_task = df_task.withColumn("run_id", F.lit(run_id))
        if model_inference_info.get("inference_task_log_table_id", None):
            df_task = df_task.withColumn("inference_id", F.lit(inference_id))
        df_task = df_task.withColumn("id", F.lit(max_id))

        write_data(
            data_path=task_log_path,
            dataframe=df_task,
            mode="overwrite",
            catalog_details = catalog_details,
            db_name = db_name,
            partition_by=None,
            is_platform_table=True,
            spark=spark,
            dbutils=dbutils,
        )
        return None


def find_integer(obj):
    if isinstance(obj, dict):  # If the input is a dictionary
        for value in obj.values():
            result = find_integer(value)  # Recursively search for an integer
            if result is not None:  # If an integer is found, return it
                return result
    elif isinstance(obj, list):  # If the input is a list
        for item in obj:
            result = find_integer(item)  # Recursively search each item for an integer
            if result is not None:  # If an integer is found, return it
                return result
    elif isinstance(obj, int):  # If the input is an integer
        return obj  # Return the integer
    return None  # If no integer is found, return None


def extract_integer(input):
    if isinstance(input, str):  # Check if the input is a string
        try:
            input = json.loads(input)  # Try to parse the JSON string
        except json.JSONDecodeError:
            return None  # Return None if the string cannot be parsed
    return find_integer(input)  # Use the find_integer function to extract the integer


def fetch_id(dbutils, transform=None):
    import json

    # Checking if task parameter variable are set
    task_info = dbutils.notebook.entry_point.getCurrentBindings()
    job_id = task_info.get("job_id", None)
    run_id = task_info.get("parent_run_id", None)

    # Getting notebook context to get task id and source information
    notebook_info = json.loads(
        dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson()
    )

    # If task parameter values are not extracted successfully, extracting job id and run id
    # from the notebook context
    if job_id == None or run_id == None:
        run_id = notebook_info["tags"]["multitaskParentRunId"]
        job_id = notebook_info["tags"]["jobId"]

    taskKey = "monitoring"
    taskDependency = ""
    if transform:
        taskDependencies = notebook_info["tags"]["taskDependencies"]
        taskDependenciesList = [x for x in taskDependencies.split('"') if "task" in x]
        if len(taskDependenciesList) == 1:
            return run_id, job_id, taskKey, taskDependenciesList[0]
        else:
            return run_id, job_id, taskKey, taskDependenciesList
    else:
        taskDependency = ""
        return run_id, job_id, taskKey, taskDependency

def get_read_data_delta_sql(data_path, spark, dbutils, table_id=None,is_platform_table=False):
    try:
        datalake_env = dbutils.widgets.get("datalake_env")
    except:
        datalake_env = "delta"

    try:
        api_endpoint = dbutils.widgets.get("tracking_base_url")
    except:
        api_endpoint = get_app_url(dbutils)

    _, vault_scope = get_env_vault_scope(dbutils)
    platform_datalake_env = dbutils.secrets.get(
        vault_scope, SECRET_MAPPING.get("platform-datalake-env", "")
    )

    env_to_read = datalake_env if not is_platform_table else platform_datalake_env

    if env_to_read == "delta" and table_id is not None:
        return uc_utils.get_read_data_delta_sql_query(
                spark=spark,
                sql=sql,
                dbutils=dbutils,
                vault_scope=vault_scope,
                api_endpoint=api_endpoint,
                headers=get_headers(vault_scope=vault_scope,dbutils=dbutils),
                table_id=table_id
            )

def read_data(data_path, spark, dbutils, table_id=None,is_platform_table=False,sql_query_string=None):

    try:
        datalake_env = dbutils.widgets.get("datalake_env")
    except:
        datalake_env = "delta"

    try:
        api_endpoint = dbutils.widgets.get("tracking_base_url")
    except:
        api_endpoint = get_app_url(dbutils)

    _, vault_scope = get_env_vault_scope(dbutils)
    platform_datalake_env = dbutils.secrets.get(
        vault_scope, SECRET_MAPPING.get("platform-datalake-env", "")
    )

    env_to_read = datalake_env if not is_platform_table else platform_datalake_env

    if env_to_read == "delta" and table_id is not None:
        print("Reading using UC UTILS",table_id,sql_query_string)
        return uc_utils.read_data(
                spark=spark,
                sql=sql,
                dbutils=dbutils,
                vault_scope=vault_scope,
                api_endpoint=api_endpoint,
                headers=get_headers(vault_scope=vault_scope,dbutils=dbutils),
                table_id=table_id,
                sql_query_string=sql_query_string
            )
    elif env_to_read == "delta":
        return utils.df_read(
            data_path=data_path, spark=spark, resource_type=env_to_read
        )
    else:
        encrypted_sa_details = dbutils.secrets.get(
            vault_scope, SECRET_MAPPING.get("gcp-service-account-encypted", "")
        )
        encryption_key = dbutils.secrets.get(
            vault_scope, SECRET_MAPPING.get("gcp-service-account-private-key", "")
        )
        bq_database_name = dbutils.secrets.get(
            vault_scope, SECRET_MAPPING.get("gcp-bq-database-name", "")
        )
        gcp_project_id = dbutils.secrets.get(
            vault_scope, SECRET_MAPPING.get("gcp-api-quota-project-id", "")
        )
        return utils.df_read(
            spark=spark,
            data_path=data_path.split(".")[-1],
            bq_database_name=bq_database_name,
            bq_project_id=gcp_project_id,
            encrypted_service_account=encrypted_sa_details,
            encryption_key=encryption_key,
            resource_type=env_to_read,
        )


def write_data(
    data_path, dataframe, mode,catalog_details,db_name, partition_by, spark, dbutils, is_platform_table=False
):

    try:
        env = dbutils.widgets.get("env")
    except:
        env = "dev"
    try:
        datalake_env = dbutils.widgets.get("datalake_env")
    except:
        datalake_env = "delta"

    _, vault_scope = get_env_vault_scope(dbutils)
    platform_datalake_env = dbutils.secrets.get(
        vault_scope, SECRET_MAPPING.get("platform-datalake-env", "")
    )
    az_container_name = str(
        dbutils.secrets.get(
            scope=vault_scope, key=SECRET_MAPPING.get("az-container-name", "")
        )
    )

    env_to_write = datalake_env if not is_platform_table else platform_datalake_env

    table_name = data_path.split(".")[-1]
    print(f"writing data in {env_to_write} for table_name: {table_name}")
    if env_to_write == "delta":
        uc_utils.write_data_in_delta(
                spark=spark,
                catalog_details=catalog_details,
                dataframe=dataframe,
                db_name=db_name,
                table_name=table_name,
                mode = mode,
                partition_by=[],
                primary_key=[],
            )
    else:
        encrypted_sa_details = dbutils.secrets.get(
            vault_scope, SECRET_MAPPING.get("gcp-service-account-encypted", "")
        )
        encryption_key = dbutils.secrets.get(
            vault_scope, SECRET_MAPPING.get("gcp-service-account-private-key", "")
        )
        bq_database_name = dbutils.secrets.get(
            vault_scope, SECRET_MAPPING.get("gcp-bq-database-name", "")
        )
        gcp_project_id = dbutils.secrets.get(
            vault_scope, SECRET_MAPPING.get("gcp-api-quota-project-id", "")
        )
        utils.df_write(
            data_path=data_path,
            dataframe=dataframe,
            mode=mode,
            bucket_name=f"{az_container_name}_{env}",
            bq_database_name=bq_database_name,
            bq_project_id=gcp_project_id,
            encrypted_service_account=encrypted_sa_details,
            encryption_key=encryption_key,
            resource_type=env_to_write,
            partition_by=partition_by,
        )


def table_already_created(data_path, spark, dbutils, is_platform_table=False):

    try:
        datalake_env = dbutils.widgets.get("datalake_env")
    except:
        datalake_env = "delta"

    _, vault_scope = get_env_vault_scope(dbutils)
    platform_datalake_env = dbutils.secrets.get(
        vault_scope, SECRET_MAPPING.get("platform-datalake-env", "")
    )
    env_to_write = datalake_env if not is_platform_table else platform_datalake_env
    if env_to_write == "delta":
        return DeltaTable.isDeltaTable(spark, data_path)
    elif env_to_write == "bigquery":
        try:
            read_data(data_path, spark, dbutils, is_platform_table).first()
            return True
        except:
            return False

@retry(
    wait=wait_exponential(min=4, multiplier=1, max=10),
    stop=(stop_after_delay(10) | stop_after_attempt(5)),
)
def table_exists(
    table_type, table_sub_type, job_id, version, project_id, dbutils,is_platform=False
):
    """
    Returns if there is a table present for the given job_id and task_id combination
    """

    params = {
        "type": table_type,
        "sub_type": table_sub_type,
        "job_id": job_id,
        "project_id": project_id,
        "version": version,
        "mode": "detail",
    }
    _ , vault_scope = get_env_vault_scope(dbutils)
    h1 = get_headers(vault_scope,dbutils)
    try:
        API_ENDPOINT = dbutils.widgets.get("tracking_base_url")
    except:
        API_ENDPOINT = get_app_url(dbutils)
    LIST_TABLES = "mlapi/tables/list"
    platform_datalake_env = dbutils.secrets.get(
        vault_scope, SECRET_MAPPING.get("platform-datalake-env", "")
    )
    response = requests.get(API_ENDPOINT + LIST_TABLES, params=params, headers=h1)
    print(
        f"\n\
    endpoint - {LIST_TABLES}\n\
    payload  - {params}\n\
    response - {response.json()}",
    )

    if response.status_code not in [200, 201]:
        raise Exception(
            f"API Error : The {LIST_TABLES} API returned {response.status_code} status code."
        )
    response = response.json()
    if response["data"]:
        if is_platform:
            for data in response["data"]:
                if data["datalake_env"] == platform_datalake_env:
                    return data
                else:
                    return None
        else:
            return response
    else:
        return None