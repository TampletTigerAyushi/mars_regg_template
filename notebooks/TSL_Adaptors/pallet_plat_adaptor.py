from TSL_Adaptors.mlcore_adaptor import get_monitor_task_log_table
from delta.tables import *
import pandas as pd
from pyspark.sql import functions as F
import json
import pandas as pd
from pyspark.sql.types import (
    DateType,
    IntegerType,
    IntegerType,
    FloatType,
)

# DONT SHOW SETTINGWITHCOPY WARNING
pd.options.mode.chained_assignment = None
from TSL_Adaptors.table_helpers import *


def df_from_sql_query(sql_query_string, spark):
    try:

        data = spark.sql(sql_query_string)
    except Exception as e:
        print(e)
        return None

    return data


def read_data_pallet_plant(
    dbutils,
    spark,
    table_details={},
    feature_columns=[],
    prediction_column=[],
    gt_column=[],
):
    table_path = table_details.get("table_path", None)
    # feature_columns = table_details.get("feature_columns", None)
    feature_columns = []
    # gt_column = table_details.get("gt_column", None)
    gt_column = ["SDD_ACTUAL_VAL"]
    prediction_column = table_details.get("prediction_column", None)
    prediction_column = ["SDD_CALC_VAL"]
    primary_keys = table_details.get("primary_keys", None)
    partition_keys = table_details.get("partition_keys", None)
    _, vault_scope = get_env_vault_scope(dbutils)
    h1 = get_headers(vault_scope, dbutils)
    batch_marker_cols = ["BATCH_ID"]
    row_marker_cols = ["RUN_ID"]
    time_stamp_cols = ["SDD_TIMESTAMP"]
    batch_size = 40

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
        print("DISPLAYING TASK LOG TABLE")
        # df_task.display()
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

        # LEFT JOIN OF Feature, Target, GT Tables
        select_sql_query = f"SELECT {','.join([f'inference_table_path.{pk}' for pk in primary_keys])} {','.join([f'inference_table_path.{fk}' for fk in feature_columns])}, inference_table_path.{prediction_column[0]} AS prediction, ground_truth_table_path.{gt_column[0]} AS y_actual, TO_DATE(inference_table_path.{time_stamp_cols[0]}) as batch_id, inference_table_path.{row_marker_cols[0]} AS id FROM delta.`{inference_table_path}` AS inference_table_path LEFT JOIN delta.`{source_table_path}` AS source_table_path ON inference_table_path.{primary_keys[0]} == source_table_path.{primary_keys[0]} LEFT JOIN delta.`{ground_truth_table_path}` AS ground_truth_table_path ON inference_table_path.{primary_keys[0]} == ground_truth_table_path.{primary_keys[0]} "
        # Zinc Inference Task Logger condition
        where_sql_query = f" WHERE inference_table_path.SDD_TAG = 'CCS' AND TO_DATE(inference_table_path.{time_stamp_cols[0]}) = (SELECT max(TO_DATE({time_stamp_cols[0]})) FROM delta.`{inference_table_path}`) AND inference_table_path.{row_marker_cols[0]} > {max_marker} AND inference_table_path.{gt_column[0]} > 0 ORDER BY inference_table_path.{row_marker_cols[0]} limit {batch_size}"
        # Filtered query
        filtered_sql_query = select_sql_query + where_sql_query

        print("**select query**")
        print(select_sql_query)
        reference_df = df_from_sql_query(select_sql_query, spark)
        print("**Donee")
        reference_df = (
            reference_df.withColumn(
                "RUN_ID", reference_df["RUN_ID"].cast(IntegerType())
            )
            .withColumn("prediction", reference_df["prediction"].cast(FloatType()))
            .withColumn("y_actual", reference_df["y_actual"].cast(FloatType()))
            .withColumn("batch_id", reference_df["batch_id"].cast(DateType()))
            .withColumn("id", reference_df["id"].cast(IntegerType()))
        )

        print("**filtered_sql_query**")
        print(filtered_sql_query)
        current_batch_df = df_from_sql_query(filtered_sql_query, spark)
        print("**Donee")
        current_batch_df = (
            current_batch_df.withColumn(
                "RUN_ID", current_batch_df["RUN_ID"].cast(IntegerType())
            )
            .withColumn("prediction", current_batch_df["prediction"].cast(FloatType()))
            .withColumn("y_actual", current_batch_df["y_actual"].cast(FloatType()))
            .withColumn("batch_id", current_batch_df["batch_id"].cast(DateType()))
            .withColumn("id", current_batch_df["id"].cast(IntegerType()))
        )

        #  Zinc Inference Task Logger
        task_logger_sql_query = f"SELECT MIN({row_marker_cols[0]}) AS start_marker , MAX({row_marker_cols[0]}) AS end_marker, MAX({row_marker_cols[0]}) AS id FROM ({filtered_sql_query})"

        print("**task_logger_sql_query**")
        print(task_logger_sql_query)
        required_entry = df_from_sql_query(task_logger_sql_query, spark)
        print("**Donee")
        required_entry = (
            required_entry.withColumn(
                "start_marker", required_entry["start_marker"].cast(IntegerType())
            )
            .withColumn("end_marker", required_entry["end_marker"].cast(IntegerType()))
            .withColumn("id", required_entry["id"].cast(IntegerType()))
        )
    elif datalake_env == "bigquery":
        pass

    return reference_df, current_batch_df, required_entry
