from TSL_Adaptors.mlcore_adaptor import get_monitor_task_log_table
from delta.tables import *
import pandas as pd
from pyspark.sql import functions as F
import json
import pandas as pd
from pyspark.sql.types import (
    IntegerType,
    FloatType,
)

# DONT SHOW SETTINGWITHCOPY WARNING
pd.options.mode.chained_assignment = None
from TSL_Adaptors.table_helpers import *


def read_data_zinc(dbutils, spark, table_details={}):

    table_path = table_details.get("table_path", None)
    table_type = table_details.get("table_type", None)

    feature_cols = []
    prediction_col = ["PMO_PRED_VALUE"]
    primary_key = ["PMO_OUTPUT_ID"]
    target_column = ["PMO_ACT_VALUE"]  # GT

    try:
        datalake_env = dbutils.widgets.get("datalake_env")
    except:
        datalake_env = "delta"

    if datalake_env == "delta":

        # get
        if table_type == "inference":
            selected_cols = ",".join(feature_cols + primary_key)

            sql_query_string = f"SELECT {selected_cols}"

            sql_query_string += f" , {','.join(prediction_col)} AS prediction "

            sql_query_string += f", {','.join(primary_key)} AS id"

            if feature_cols == []:
                sql_query_string += ", '1' AS feature "

            sql_query_string += f"CAST(prediction AS Float) "

            sql_query_string += f"FROM delta.`{table_path}`"

        # primary key column should be a sortable column , idealy an integer column
        elif table_type == "inference_task_logger":
            sql_query_string = f"SELECT MIN({'.'.join(primary_key)}) AS start_marker "

            sql_query_string += f", MAX({'.'.join(primary_key)}) AS end_marker "

            sql_query_string += f", MIN({'.'.join(primary_key)}) AS id "

            sql_query_string += f"FROM delta.`{table_path}` "

            where_rules_str = (
                "WHERE PMO_PRED_STARTDT > PMO_CRT_ON AND PMO_ACT_VALUE > 0"
            )
            # condtion to get the start marker and end marker for current batch
            if where_rules_str:
                sql_query_string += where_rules_str

        elif table_type == "ground_truth":
            sql_query_string = f"SELECT {','.join(primary_key + target_column)} FROM delta.`{table_path}`"

            where_rules_str = "WHERE PMO_ACT_VALUE > 0"
            if where_rules_str:
                sql_query_string += where_rules_str

        elif table_type == "source":

            sql_query_string = f"SELECT {','.join(primary_key + feature_cols)}"

            if feature_cols == []:
                sql_query_string += ", '1' AS feature "

            sql_query_string += f"FROM delta.`{table_path}`"

        print(sql_query_string)

    return df_from_sql_query(sql_query_string, spark)


def df_from_sql_query(sql_query_string, spark):
    try:

        data = spark.sql(sql_query_string)
        # connection = sql.connect(server_hostname=str(Databricks_host),
        #                         http_path=DATABRICKS_HTTP_PATH,
        #                         access_token= DATABRICKS_TOKEN)
        # cursor = connection.cursor()
        # data = cursor.execute(sql_query_string)
        # result = data.fetchall()
    except Exception as e:
        print(e)
        return None

    return data


def read_data_zinc2(
    dbutils,
    spark,
    table_details={},
    feature_columns=[],
    prediction_column=[],
    gt_column=[],
):
    table_path = table_details.get("table_path", None)
    feature_columns = table_details.get("feature_columns", None)
    gt_column = table_details.get("gt_column", None)
    prediction_column = table_details.get("prediction_column", None)
    prediction_column = ["PMO_PRED_VALUE"]
    primary_keys = table_details.get("primary_keys", None)
    partition_keys = table_details.get("partition_keys", None)
    _, vault_scope = get_env_vault_scope(dbutils)
    h1 = get_headers(vault_scope, dbutils)
    batch_marker_cols = ["PMO_PRED_STARTDT", "PMO_CRT_ON"]
    row_marker_cols = ["PMO_OUTPUT_ID"]
    batch_size = 1

    try:
        datalake_env = dbutils.widgets.get("datalake_env")
    except:
        datalake_env = "delta"

    if datalake_env == "delta":
        # Delta Table Paths
        inference_table_path = table_path["inference_table_path"]
        source_table_path = table_path["source_table_path"]
        ground_truth_table_path = table_path["ground_truth_table_path"]
        monitring_task_log_path = table_path["monitring_task_log_path"]

        # Fetch the last row that was monitored
        model_inference_info = dbutils.widgets.get("model_inference_info")
        if isinstance(model_inference_info, str):
            model_inference_info = json.loads(model_inference_info)

        # Fetching task log table for the monitoring subtype
        df_task = get_monitor_task_log_table(dbutils, spark, monitring_task_log_path)
        if df_task is not None:
            print("DISPLAYING TASK LOG TABLE")
            df_task.display()
        # Determining max marker
        if df_task == None:
            max_marker = 0
        elif model_inference_info.get("inference_task_log_table_id", None):
            max_marker1 = int(df_task.select(F.max("inference_id")).collect()[0][0])
            try:
                max_marker2 = int(df_task.select(F.max("end_marker")).collect()[0][0])
            except:
                max_marker2 = 1
            max_marker = max(max_marker1, max_marker2)
        else:
            max_marker = int(df_task.select(F.max("end_marker")).collect()[0][0])
        print("MAX_MARKER", max_marker)

        selected_columns = primary_keys + feature_columns
        # LEFT JOIN OF Feature, Target, GT Tables
        select_sql_query = f"SELECT {','.join([f'inference_table_path.{sk}' for sk in selected_columns])} , inference_table_path.{prediction_column[0]} AS prediction, ground_truth_table_path.{gt_column[0]} AS y_actual, inference_table_path.{row_marker_cols[0]} AS id FROM delta.`{inference_table_path}` AS inference_table_path LEFT JOIN delta.`{source_table_path}` AS source_table_path ON inference_table_path.{primary_keys[0]} == source_table_path.{primary_keys[0]} LEFT JOIN delta.`{ground_truth_table_path}` AS ground_truth_table_path ON inference_table_path.{primary_keys[0]} == ground_truth_table_path.{primary_keys[0]} "
        # Zinc Inference Task Logger condition
        where_sql_query = f" WHERE inference_table_path.{row_marker_cols[0]} > {max_marker} AND inference_table_path.{gt_column[0]} > 0 ORDER BY inference_table_path.{row_marker_cols[0]} limit {batch_size}"
        # Filtered query
        filtered_sql_query = select_sql_query + where_sql_query

        print(select_sql_query)
        reference_df = df_from_sql_query(select_sql_query, spark)
        reference_df = (
            reference_df.withColumn(
                "PMO_OUTPUT_ID", reference_df["PMO_OUTPUT_ID"].cast(IntegerType())
            )
            .withColumn("prediction", reference_df["prediction"].cast(FloatType()))
            .withColumn("y_actual", reference_df["y_actual"].cast(FloatType()))
            .withColumn("id", reference_df["id"].cast(IntegerType()))
        )

        print(filtered_sql_query)
        current_batch_df = df_from_sql_query(filtered_sql_query, spark)
        current_batch_df = (
            current_batch_df.withColumn(
                "PMO_OUTPUT_ID", current_batch_df["PMO_OUTPUT_ID"].cast(IntegerType())
            )
            .withColumn("prediction", current_batch_df["prediction"].cast(FloatType()))
            .withColumn("y_actual", current_batch_df["y_actual"].cast(FloatType()))
            .withColumn("id", current_batch_df["id"].cast(IntegerType()))
        )

        #  Zinc Inference Task Logger
        task_logger_sql_query = f"SELECT MIN({row_marker_cols[0]}) AS start_marker , MAX({row_marker_cols[0]}) AS end_marker FROM ({filtered_sql_query})"

        print(task_logger_sql_query)
        required_entry = df_from_sql_query(task_logger_sql_query, spark)
        required_entry = required_entry.withColumn(
            "start_marker", required_entry["start_marker"].cast(IntegerType())
        ).withColumn("end_marker", required_entry["end_marker"].cast(IntegerType()))

    elif datalake_env == "bigquery":
        pass

    return reference_df, current_batch_df, required_entry
