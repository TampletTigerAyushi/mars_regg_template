import os
import ast
import requests
from requests.structures import CaseInsensitiveDict
import time
import requests
from datetime import datetime
from TSL_Adaptors.mlcore_adaptor import get_inference_batch, get_monitor_task_log_table
from delta.tables import *
import pandas as pd
import numpy as np
from pyspark.sql import types as DT, functions as F, Window
import ast
import traceback
import sys
import json
import pandas as pd
from tenacity import retry, stop_after_delay, stop_after_attempt, wait_exponential
import re
import copy
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    DateType,
    IntegerType,
    LongType,
    FloatType,
)
from pyspark.sql import types as T
from pyspark.sql import Row
from pyspark.sql.functions import unix_timestamp

# DONT SHOW SETTINGWITHCOPY WARNING
pd.options.mode.chained_assignment = None
from monitoring.utils import utils
from TSL_Adaptors.table_helpers import *


def read_data_cldc(
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
    primary_keys = table_details.get("primary_keys", None)
    partition_keys = table_details.get("partition_keys", None)
    _, vault_scope = get_env_vault_scope(dbutils)
    h1 = get_headers(vault_scope, dbutils)
    prediction_column = ["CLDC_PRED_LOAD_TS"]
    batch_marker_cols = ["KPI_OCR_DT"]
    row_marker_cols = ["KPI_OCR_DT"]
    batch_size = 96

    try:
        datalake_env = dbutils.widgets.get("datalake_env")
    except:
        datalake_env = "delta"

    if datalake_env == "delta":
        # Delta Table Paths
        inference_table_path = table_path["inference_table_path"] #load_pred
        source_table_path = table_path["source_table_path"] #prm_values
        ground_truth_table_path = table_path["ground_truth_table_path"] #prm_values
        inference_task_logger_path = table_path["inference_task_log_path"] #prm_master
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

        # Select Query for Reference Data
        selected_columns = primary_keys + feature_columns
        select_sql_query = f"""Select DISTINCT {','.join([f'source_table_path.{sk}' for sk in selected_columns])} , source_table_path.{gt_column[0]} AS y_actual
                            FROM  delta.`{source_table_path}` AS source_table_path
                            JOIN delta.`{inference_task_logger_path}` AS prm_master
                            ON
                            source_table_path.KPI_ID = prm_master.KPM_ID AND prm_master.KPM_ACT_TAG = 'A'
                            AND prm_master.KPM_DESC = 'CALC' AND prm_master.KPM_UOM = 'MW'
                            AND prm_master.KPM_LOC_ID= 'L100' AND prm_master.KPM_ORDER_ID = '1'
                            order by source_table_path.KPI_OCR_DT desc"""

        print(select_sql_query)
        reference_df = df_from_sql_query(select_sql_query, spark)
        reference_pd_df = reference_df.toPandas()
        reference_pd_df = reference_pd_df.set_index(["KPI_OCR_DT"])
        reference_pd_df = reference_pd_df[reference_pd_df["y_actual"] > 0]
        reference_pd_df["y_actual"] = reference_pd_df["y_actual"].abs()
        reference_pd_df = reference_pd_df.resample("15MIN").mean().reset_index()
        reference_df = spark.createDataFrame(
            reference_pd_df
        )  # contains, primary_key & y_actual

        # Need to join with below table, to get predictions
        inference_sql_query = f"Select * from delta.`{inference_table_path}`"
        inference_df = df_from_sql_query(inference_sql_query, spark)
        inf_columns = primary_keys + prediction_column
        inference_df = inference_df.select(*inf_columns)
        reference_df2 = inference_df.join(reference_df, "KPI_OCR_DT", "left")
        # TODO: Limit the reference data to certain no. of rows
        reference_df2 = reference_df2.withColumnRenamed(
            prediction_column[0], "prediction"
        ).orderBy(primary_keys[0])
        # Need row marker as ID after resampling
        # FINAL REFERENCE DF BELOW
        reference_df2 = reference_df2.withColumn("id", unix_timestamp("KPI_OCR_DT"))
        reference_df2 = reference_df2.dropna(how="any")

        # Current Table Logic, we will slice the reference_table to get current_df
        if max_marker == 1:
            condition2 = (F.col("y_actual").isNotNull()) & (F.col("y_actual") > 0)
            # FINAL CURRENT DF BELOW
            current_df = reference_df2.filter(condition2).limit(batch_size)
        else:
            condition1 = F.col("id") > max_marker
            condition2 = (F.col("y_actual").isNotNull()) & (F.col("y_actual") > 0)
            # FINAL CURRENT DF BELOW
            current_df = (
                reference_df2.orderBy("id")
                .filter(condition1 & condition2)
                .limit(batch_size)
            )

        # Inference_Task_Logger Logic
        start_marker = current_df.agg({"id": "min"}).collect()[0][0]
        end_marker = current_df.agg({"id": "max"}).collect()[0][0]
        id = end_marker
        print("START_MARKER", start_marker)
        print("END_MARKER", end_marker)
        print("ID", id)
        # data = [(start_marker, end_marker, id)]
        # rows = [Row(start_marker=row[0], end_marker=row[1], id=row[2]) for row in data]
        # columns = ["start_marker", "end_marker", "id"]
        # # FINAL TASK LOG TABLE BELOW
        schema = StructType(
            [
                StructField("start_marker", IntegerType(), False),
                StructField("end_marker", IntegerType(), False),
                StructField("id", IntegerType(), False),
            ]
        )
        inference_task_log_df = spark.createDataFrame(
            [(start_marker, end_marker, id)], schema=schema
        )
        return reference_df2, current_df, inference_task_log_df


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
