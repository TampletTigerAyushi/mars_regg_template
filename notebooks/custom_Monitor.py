# Databricks notebook source
# MAGIC %pip install sparkmeasure
# MAGIC %pip install databricks-sql-connector

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from sparkmeasure import StageMetrics
from sparkmeasure import TaskMetrics

taskmetrics = TaskMetrics(spark)
stagemetrics = StageMetrics(spark)

taskmetrics.begin()
stagemetrics.begin()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Imports

# COMMAND ----------

import os
import ast
import requests
from requests.structures import CaseInsensitiveDict
import time
import requests
from datetime import datetime
from delta.tables import *
import pandas as pd
import numpy as np
import pyspark
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
)
from pyspark.sql import types as T
from pyspark.sql.types import *
from databricks import sql
from monitoring.utils.secret_mapping import SECRET_MAPPING, CONFIGS
from monitoring.utils.vault_scope import VAULT_SCOPE

# DONT SHOW SETTINGWITHCOPY WARNING
pd.options.mode.chained_assignment = None
from monitoring.utils import utils
from monitoring.utils import uc_utils
# Importing TSL Adaptor scripts
from TSL_Adaptors.adaptor import get_monitoring_tables

# COMMAND ----------

# MAGIC %md
# MAGIC ## Utility Functions

# COMMAND ----------


def get_env_vault_scope():
    """
    Returns env and vault scope
    """
    try:
        env = dbutils.widgets.get("deployment_env")
    except:
        env = (
            dbutils.notebook.entry_point.getDbutils()
            .notebook()
            .getContext()
            .notebookPath()
            .get()
        ).split("/")[2]

    return env, VAULT_SCOPE.get(env, {}).get("client_name", "")



def get_access_tokens(client_id, scope, client_secret, vault_scope):
    """
    Returns a bearer token
    """

    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/x-www-form-urlencoded"
    data = {}
    data["client_id"] = client_id
    data["grant_type"] = "client_credentials"
    data["scope"] = scope
    data["client_secret"] = client_secret
    tenant_id = secrets_object.get("az-directory-tenant", "")
    url = "https://login.microsoftonline.com/" + tenant_id + "/oauth2/v2.0/token"
    resp = requests.post(url, headers=headers, data=data).json()
    token = resp["access_token"]
    token_string = "Bearer" + " " + token
    return token_string


def get_app_url():
    """
    Returns env and vault scope
    """
    print("Fetching API_ENDPOINT from secrets.")
    env, vault_scope = get_env_vault_scope()
    API_ENDPOINT = ""
    if env in ["dev", "qa"]:
        API_ENDPOINT = (
            secrets_object.get(f"az-app-service-{env}-url-2", "") + "/"
        )
    else:
        API_ENDPOINT = (
            secrets_object.get(f"az-app-service-url", "") + "/"
        )
    return API_ENDPOINT


def get_headers(vault_scope):
    """
    Returns API headers
    """
    h1 = CaseInsensitiveDict()
    client_id = secrets_object.get("az-api-client-id", "")
    scope = client_id + "/.default"
    client_secret = secrets_object.get("az-api-client-secret", "")
    h1["Authorization"] = get_access_tokens(
        client_id, scope, client_secret, vault_scope
    )
    h1["Content-Type"] = "application/json"
    return h1


def get_cluster_info(spark):
    p = "spark.databricks.clusterUsageTags."
    conf = spark.sparkContext.getConf().getAll()
    conf_dict = {k.replace(p, ""): v for k, v in conf if k.startswith(p)}

    return conf_dict


def fetch_secrets_from_dbutils(dbutils, message_logs=[]):
    _, vault_scope = get_env_vault_scope()
    secrets_object = {}
    for secret_key, secret_value in SECRET_MAPPING.items():
        try:
            secrets_object[secret_key] = dbutils.secrets.get(
                scope=vault_scope, key=secret_value
            )
        except Exception as e:
            utils.log(
                f"Error fetching secret for '{secret_value} using dbutils': {e}",
                message_logs,
                "warning",
            )

            # fetch the secret from config file
            secrets_object[secret_key] = CONFIGS.get(secret_value, "")

    return secrets_object


def get_params(name):
    import json

    param = dbutils.widgets.get(name)
    param = param.replace("'", '"')
    param = json.loads(param)
    return param


def get_monitoring_config(deployment_master_id):
    """
    Returns model monitoring configs as per the deployment master id
    """
    url = os.path.join(API_ENDPOINT, "mlapi/monitoring_configs/get")
    url = url + f"?deployment_master_id={deployment_master_id}"
    payload = {}
    response = requests.get(url=url, headers=h1, data=payload).json()
    utils.log(
        f"\n\
    endpoint - mlapi/monitoring_configs/get\n\
    response - {response}",
        message_run,
    )
    return response["data"]


def write_metrics_to_delta(metric_json, kind):
    """
    Takes in metric json and writes delta at a dedicated metrics delta path for the resp kind of metric list
    """
    try:
        # Determining target table path
        if kind != "performance_drift":
            metric_table_path = f"{env}_monitoring_{job_id}_data_model_drift"
            table_type = "Monitoring_Output"
            table_sub_type = "Data_Model"
        else:
            metric_table_path = f"{env}_monitoring_{job_id}_performance_drift"
            table_type = "Monitoring_Output"
            table_sub_type = "Performance"
        utils.log(f"METRIC_TABLE_PATH : {metric_table_path}", message_run)
        # Fetching target table schema
        schema = get_metric_table_schema(kind)

        # Empty string treatment for delta compatibility
        metric_json = json_encode_nans(metric_json)
        metric_json = delta_encode_str(metric_json)
        restructured_json = restructure_metric_json(metric_json, kind)

        colnames = [
            k
            for k, v in restructured_json[0].items()
            if k not in ["data", "monitoring_algo"]
        ]
        utils.log(f"COLNAMES - {colnames}", message_run)
        # Converting metric json into a dataframe
        pd_metric_df = pd.json_normalize(restructured_json, "data", colnames)

        # Ordering columns of the pandas daraframe as per the target table schema
        pd_metric_df = pd_metric_df[[entry.name for entry in schema.fields]]

        # If null exist in is_drift, pandas converts this int column into float, which breaks delta
        # Checking if this scenario is present, and typecasting to handle the scenario accordingly
        if (
            pd_metric_df.dtypes["is_drift"] != "int"
            and pd_metric_df["is_drift"].isna().sum() > 0
        ):
            pd_metric_df["is_drift"] = (
                pd_metric_df["is_drift"].astype("Int64").fillna(-1)
            )

        # Converting pandas dataframe into a pyspark dataframe
        metric_df = spark.createDataFrame(pd_metric_df, schema=schema)

        # Adding the date literal column
        metric_df = metric_df.withColumn("date", F.lit(date))
        metric_df = metric_df.withColumn("date", F.to_date(F.col("date"), "MM-dd-yyyy"))
        metric_df = metric_df.withColumn(
            "timestamp",
            F.expr("reflect('java.lang.System', 'currentTimeMillis')").cast("long"),
        )

        if kind != "performance_drift":
            metric_df = metric_df.withColumn(
                "modelling_task_name", F.lit(modelling_task_type)
            )

        # print("METRIC_DF : ", message_run)
        metric_df.display()

        # Saving the table into the target path
        table_exist_responce = table_exists(table_type,table_sub_type,job_id,version,project_id)
        if table_exist_responce:
            # Check the existing metrics data and take action based on below condition:
            # If "id" column is present, start the id from max_id + 1
            # If "id" column is not present, merge existing and new df and then restart the id from 1
            # If the table does not exists, simply start the id from 1
            existing_metric_table_data = read_data("",table_id=table_exist_responce["data"][0]["table_id"])
            if "id" in existing_metric_table_data.columns:
                max_id = existing_metric_table_data.select(F.max("id")).collect()[0][0]
                w = Window.orderBy(F.lit(0))
                metric_df = metric_df.withColumn("id", F.row_number().over(w) + max_id)
            else:
                metric_df = existing_metric_table_data.union(metric_df)
                w = Window.orderBy(F.monotonically_increasing_id())
                metric_df = metric_df.withColumn("id", F.row_number().over(w))
            try:
                write_data(
                    data_path=metric_table_path,
                    dataframe=metric_df,
                    mode="append",
                    db_name = db_name,
                    table_name = metric_table_path,
                    partition_by=["date"],
                )
            except:
                write_data(
                    data_path=metric_table_path,
                    dataframe=metric_df,
                    mode="overwrite",
                    db_name = db_name,
                    table_name = metric_table_path,
                    partition_by=["date"],
                )
        else:
            w = Window.orderBy(F.monotonically_increasing_id())
            metric_df = metric_df.withColumn("id", F.row_number().over(w))
            if kind not in ["feature_drift","performance_drift"]:
                mode = "append"
            else:
                mode = "overwrite"
            write_data(
                data_path=metric_table_path,
                dataframe=metric_df,
                mode=mode,
                db_name = db_name,
                table_name = metric_table_path,
                partition_by=["date"],
            )

        # Write Aggregate Data to the table
        if kind != "performance_drift":
            mode = (
                "overwrite"
                if not table_already_created(
                    table_name=aggregated_dd_table_name,
                    schema_name=platform_db_name,
                    catalog_name=uc_catalog_name
                )
                else "append"
            )
            write_data(
                data_path=aggregated_dd_drift_path,
                dataframe=metric_df,
                mode=mode,
                db_name = platform_db_name,
                partition_by=[
                    "project_id",
                    "version",
                    "deployment_master_id",
                    "monitoring_algo_id",
                ],
                table_name = aggregated_dd_table_name,
                is_platform_table=True,
            )
        else:
            mode = (
                "overwrite"
                if not table_already_created(
                    table_name=aggregated_pd_table_name,
                    schema_name=platform_db_name,
                    catalog_name=uc_catalog_name,
                )
                else "append"
            )
            write_data(
                data_path=aggregated_performance_drift_path,
                dataframe=metric_df,
                mode=mode,
                db_name = platform_db_name,
                table_name = aggregated_pd_table_name,
                partition_by=["project_id", "version", "deployment_master_id"],
                is_platform_table=True,
            )

        # Register table to Hive with the datalake env is delta
        # if datalake_env.lower() == "delta":
        #     register_delta_as_hive(
        #         db_name, f"{kind}_{job_id}", metric_table_path, spark
        #     )

    except Exception as e:
        traceback.print_exc()
        # FIXME:
        # Commenting out this as it is causing inference_metadata to not be written
        # on monitorkits end
        # throw_exception(e)


def replace_empty_string_with_none(local_feat_entry):
    """
    If value of a key is an empty string, replaces it with None for delta datatype compatibility
    """
    for local_feat_entry_k, local_feat_entry_v in local_feat_entry.items():
        if local_feat_entry_v == "":
            local_feat_entry[local_feat_entry_k] = None
    return local_feat_entry


def restructure_metric_json(metric_json, kind):
    """
    Restructures generated metric json into delta template format
    """
    copy_metric_json = copy.deepcopy(metric_json)
    for entry in copy_metric_json:
        # restructing data object
        data = entry.get("data", {})
        restructured_data = []
        if kind == "feature_drift":
            for feat_entry, feat_entry_obj in data.items():
                local_feat_entry = {}
                if isinstance(feat_entry_obj, dict):
                    local_feat_entry["feature_attribute"] = (
                        feat_entry
                        if entry.get("monitoring_sub_type", "") == "feature_level"
                        else "overall"
                    )
                    local_feat_entry["is_drift"] = feat_entry_obj.get("is_drift", None)
                    local_feat_entry["stat_val"] = feat_entry_obj.get("stat_val", None)
                    local_feat_entry["p_val"] = feat_entry_obj.get("p_val", None)
                    local_feat_entry = replace_empty_string_with_none(local_feat_entry)
                    restructured_data.append(local_feat_entry)

        elif kind in ["target_drift", "concept_drift"]:
            local_feat_entry = {}
            local_feat_entry["feature_attribute"] = "overall"
            local_feat_entry["is_drift"] = data.get("is_drift", None)
            local_feat_entry["stat_val"] = data.get("stat_val", None)
            local_feat_entry["p_val"] = data.get("p_val", None)
            local_feat_entry = replace_empty_string_with_none(local_feat_entry)
            restructured_data.append(local_feat_entry)

        else:
            for feat_entry, feat_entry_obj in data.items():
                local_feat_entry = {}
                if isinstance(feat_entry_obj, dict):
                    local_feat_entry["feature_attribute"] = feat_entry
                    local_feat_entry["is_drift"] = feat_entry_obj.get("is_drift", None)
                    local_feat_entry["value"] = feat_entry_obj.get("value", None)
                else:
                    local_feat_entry["feature_attribute"] = "overall"
                    local_feat_entry["is_drift"] = data.get("is_drift", None)
                    local_feat_entry["value"] = data.get("value", None)
                local_feat_entry = replace_empty_string_with_none(local_feat_entry)
                restructured_data.append(local_feat_entry)

        if "feature_attribute" not in data.keys():
            data["feature_attribute"] = "overall"

        entry["data"] = restructured_data if len(restructured_data) != 0 else [data]

        # restructing monitoring algo
        moni_algo = entry.get("monitoring_algo", {})
        if kind != "performance_drift":
            entry["monitoring_algo_id"] = moni_algo.get("algo_id")
        entry["modelling_task_name"] = moni_algo.get("modelling_task_name")
        entry["monitoring_algo_name"] = moni_algo.get("algo_name")
        try:
            del entry["monitoring_algo"]
        except:
            pass
    return copy_metric_json


def check_if_upstream_job_active(job_id):
    """
    Checks if the job is active
    """
    try:
        if isinstance(job_id, list):
            job_id = ",".join(job_id)

        url = os.path.join(API_ENDPOINT, JOB_LOGS)
        url = url + f"?job_id={job_id}"
        headers = {"Authorization": h1["Authorization"]}
        response = requests.request("GET", url, headers=headers, data={})

        response = response.json()
        data = response.get("data", [])
        if not data:
            data = []

        status_of_jobs = [entry.get("status", "active") for entry in data]
        if "archived" not in status_of_jobs:
            return True
        else:
            return False

    except Exception as e:
        utils.log(str(e), message_run)
        return True


def declare_job_as_successful(
    no_data=False,
    no_monitor_config=False,
    no_inference_data=False,
    upstream_inactive=False,
    skipped_due_to_gt=[],
    drifted_monitor_types=[],
    aggregated_is_drift=False,
    processed_records="yes",
):
    """
    Setting the Monitor job status as successful by updating Job Runs Log and Job Tasks Log
    """
    stagemetrics.end()
    taskmetrics.end()

    stage_Df = stagemetrics.create_stagemetrics_DF("PerfStageMetrics")
    task_Df = taskmetrics.create_taskmetrics_DF("PerfTaskMetrics")

    aggregate_compute_metrics = (
        stagemetrics.aggregate_stagemetrics_DF()
        .select("executorCpuTime", "peakExecutionMemory")
        .collect()[0]
        .asDict()
    )

    aggregate_compute_metrics["executorCpuTime"] = (
        aggregate_compute_metrics["executorCpuTime"] / 1000
        if aggregate_compute_metrics["executorCpuTime"]
        else 0
    )
    aggregate_compute_metrics["peakExecutionMemory"] = (
        aggregate_compute_metrics["peakExecutionMemory"] / (1024 * 1024)
        if aggregate_compute_metrics["peakExecutionMemory"]
        else 0
    )

    compute_metrics = {
        "stagemetrics": stage_Df.rdd.map(lambda row: row.asDict()).collect(),
        "taskmetrics": task_Df.rdd.map(lambda row: row.asDict()).collect(),
    }

    end_time = str(int(time.time() * 1000000))
    utils.log("final logs into collection", message_run)
    job_id, run_id, task_run_id, taskKey = utils.get_job_details(dbutils)
    model_inference_info = dbutils.widgets.get("model_inference_info")
    if isinstance(model_inference_info, str):
        model_inference_info = json.loads(model_inference_info)
    project_id, version = (
        model_inference_info["project_id"],
        model_inference_info["version"],
    )
    created_by_id, created_by_name = (
        model_inference_info["created_by_id"],
        model_inference_info["created_by_name"],
    )

    # Converting aggregated is drift flag from boolean to string for API compatibility
    if aggregated_is_drift:
        aggregated_is_drift = "yes"
    else:
        aggregated_is_drift = "no"

    # Fetching headers
    h1 = get_headers(vault_scope)

    if no_data:
        message_run.append(
            {
                "time": str(end_time),
                "message": "Monitor Job with job id: "
                + str(job_id)
                + " and run id: "
                + str(run_id)
                + " had no data to monitor.",
            }
        )
        message_task.append(
            {
                "time": str(end_time),
                "message": taskKey
                + " within job id: "
                + str(job_id)
                + " and run id: "
                + str(run_id)
                + " had no data to monitor.",
            }
        )
    elif no_monitor_config:
        message_run.append(
            {
                "time": str(end_time),
                "message": "Monitor Job with job id: "
                + str(job_id)
                + " and run id: "
                + str(run_id)
                + " did not find monitoring configuration.",
            }
        )
        message_task.append(
            {
                "time": str(end_time),
                "message": taskKey
                + " within job id: "
                + str(job_id)
                + " and run id: "
                + str(run_id)
                + " did not find monitoring configuration.",
            }
        )

    elif no_inference_data:
        message_run.append(
            {
                "time": str(end_time),
                "message": " Monitor Job with  job id: "
                + str(job_id)
                + " and run id: "
                + str(run_id)
                + " cannot proceed before at least one successful run of model inference gets completed.",
            }
        )
        message_task.append(
            {
                "time": str(end_time),
                "message": taskKey
                + " within  job id: "
                + str(job_id)
                + " and run id: "
                + str(run_id)
                + " cannot proceed before at least one successful run of model inference gets completed.",
            }
        )
    elif upstream_inactive:
        message_run.append(
            {
                "time": str(end_time),
                "message": f"The data prep deployment pipeline or model inference pipeline, \
                            on which the model inference job with job id: {job_id} and run id: {run_id} has been triggered, is inactive.",
            }
        )
        message_task.append(
            {
                "time": str(end_time),
                "message": f"The data prep deployment pipeline or model inference pipeline, \
                            on which the model inference job with job id: {job_id} and run id: {run_id} has been triggered, is inactive.",
            }
        )
    elif len(null_gt_skipped_types) > 0:
        all_skipped_types = ", ".join(null_gt_skipped_types)
        message = (
            f"Monitor Job with job id: {job_id} and run id: {run_id} could not process {all_skipped_types} as there are null values in the ground truth "
            "for some of the features."
        )
        message_run.append(
            {
                "time": str(end_time),
                "message": message,
            }
        )
        message_task.append(
            {
                "time": str(end_time),
                "message": message,
            }
        )
    else:
        message_run.append(
            {
                "time": str(end_time),
                "message": "Monitor Job with job id: "
                + str(job_id)
                + " and run id: "
                + str(run_id)
                + " is finished successfully.",
            }
        )
        message_task.append(
            {
                "time": str(end_time),
                "message": taskKey
                + " within job id: "
                + str(job_id)
                + " and run id: "
                + str(run_id)
                + " success.",
            }
        )

    log_data = {
        "project_id": project_id,
        "version": str(version),
        "job_id": str(job_id),
        "run_id": str(run_id),
        "task_id": str(task_run_id),
        "end_time": str(end_time),
        "status": "success",
        "message": message_run,
        "updated_by_id": created_by_id,
        "updated_by_name": created_by_name,
        "created_by_id": created_by_id,
        "created_by_name": created_by_name,
        "job_type": "Monitor",
        "drift_monitor_type": drifted_monitor_types,
        "is_drift": aggregated_is_drift,
        "processed_records": processed_records,
        "cpu": str(aggregate_compute_metrics.get("executorCpuTime", "NA")),
        "ram": str(aggregate_compute_metrics.get("peakExecutionMemory", "NA")),
        "cluster_info": get_cluster_info(spark),
        "compute_metrics": compute_metrics,
        "run_notebook_url": run_notebook_url,
        "deployment_master_id": deployment_master_id,
    }
    # Log the payload excluding large fields
    log_payload = {
        key: value
        for key, value in log_data.items()
        if key not in ["compute_metrics", "cluster_info", "message"]
    }
    print(f"API URL : {JOB_RUNS_UPDATE}. Payload : {log_payload}")

    response = requests.put(API_ENDPOINT + JOB_RUNS_UPDATE, json=log_data, headers=h1)
    print(
        f"\n\
    Logging task:\n\
    endpoint - {JOB_RUNS_UPDATE}\n\
    status   - {response}\n\
    response - {response.text}"
    )

    utils.save_sparkmeasure_aggregated_tables(
        project_name=project_name,
        job_run_update_payload=log_data,
        compute_usage_metrics=aggregate_compute_metrics,
        stagemetrics=stagemetrics,
        taskmetrics=taskmetrics,
        api_endpoint=API_ENDPOINT,
        headers=h1,
        spark=spark,
        dbutils=dbutils,
    )

    task_log_data = {
        "job_id": str(job_id),
        "run_id": str(run_id),
        "task_id": str(task_run_id),
        "end_time": str(end_time),
        "status": "success",
        "message": message_task,
        "updated_by_id": created_by_id,
        "updated_by_name": created_by_name,
        "created_by_id": created_by_id,
        "created_by_name": created_by_name,
        "job_type": "Monitor",
        "run_notebook_url": run_notebook_url,
        "cpu": str(aggregate_compute_metrics.get("executorCpuTime", "NA")),
        "ram": str(aggregate_compute_metrics.get("peakExecutionMemory", "NA")),
        "cluster_info": get_cluster_info(spark),
        "compute_metrics": compute_metrics,
    }

    # Log the payload excluding large fields
    log_payload = {
        key: value
        for key, value in task_log_data.items()
        if key not in ["compute_metrics", "cluster_info", "message"]
    }
    print(f"API URL : {JOB_TASK_UPDATE}. Payload : {log_payload}")

    response = requests.put(
        API_ENDPOINT + JOB_TASK_UPDATE, json=task_log_data, headers=h1
    )
    print(
        f"\n\
    Logging task:\n\
    endpoint - {JOB_TASK_UPDATE}\n\
    status   - {response}\n\
    response - {response.text}"
    )

    # Exiting the notebook
    try:
        dbutils.notebook.exit("Job is Successful!")
    except:
        pass


def generate_run_notebook_url(job_id, run_id):
    """
    Generates the databricks job run notebook url in runtime
    """
    workspace_url = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
    workspace_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterOwnerOrgId")
    run_notebook_url = f"{workspace_url}/?o={workspace_id}#job/{job_id}/run/{run_id}"
    return run_notebook_url


@retry(
    wait=wait_exponential(min=4, multiplier=1, max=10),
    stop=(stop_after_delay(10) | stop_after_attempt(5)),
)
def table_exists(
    table_type, table_sub_type, job_id, version, project_id, is_platform=False
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
    headers = get_headers(vault_scope)
    response = requests.get(API_ENDPOINT + LIST_TABLES, params=params, headers=h1)
    utils.log(
        f"\n\
    endpoint - {LIST_TABLES}\n\
    payload  - {params}\n\
    response - {response.json()}",
        message_run,
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


def tables_add(data):
    """
    Calls tables add API with the entered data
    """
    h1 = get_headers(vault_scope)
    response = requests.post(API_ENDPOINT + TABLES_ADD, json=data, headers=h1)
    utils.log(
        f"\n\
    endpoint - {TABLES_ADD}\n\
    payload  - {data}\n\
    response - {response}",
        message_run,
    )

    # Raise exception if table_add is failing
    if response.status_code not in [200, 201]:
        try:
            response_message = response.json()
        except:
            response_message = ""

        raise Exception(
            f"API Error : The {TABLES_ADD} API returned {response.status_code} status code. Response : {response_message}"
        )
    return response


def get_table_dbfs_path(table_type, table_sub_type,catalog_details,table_id=None):
    """
    Returns templated dbfs path as per the table type and sub type
    """
    platform_cloud_provider = catalog_details.get("platform_cloud_provider", "hive")
    if table_type == "Monitoring_Output":
        if table_sub_type == "Data_Model":
            if platform_cloud_provider == "databricks_uc":
                catalog_name = catalog_details.get("catalog_name", "")
                table_name = f"{env}_monitoring_{job_id}_data_model_drift"
                path = f"{catalog_name}.{db_name}.{table_name}"
            elif datalake_env.lower() == "delta":
                table_name = f"{env}_monitoring_{job_id}_data_model_drift"
                output_path = f"{db_name}.{table_name}"
                path = spark.sql(f"desc formatted {output_path}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]
            else:
                path = f"{env}_monitoring_{job_id}_data_model_drift"
        else:
            if platform_cloud_provider == "databricks_uc":
                catalog_name = catalog_details.get("catalog_name", "")
                table_name = f"{env}_monitoring_{job_id}_performance_drift"
                path = f"{catalog_name}.{db_name}.{table_name}"
            elif datalake_env.lower() == "delta":
                table_name = f"{env}_monitoring_{job_id}_performance_drift"
                output_path = f"{db_name}.{table_name}"
                path = spark.sql(f"desc formatted {output_path}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]
            else:
                path = f"{env}_monitoring_{job_id}_performance_drift"

    elif table_type == "Task_Log":
        if platform_cloud_provider == "databricks_uc":
            catalog_name = catalog_details.get("catalog_name", "")
            table_name = f"{env}_{job_id}_task_log_table"
            path = f"{catalog_name}.{db_name}.{table_name}"
        elif datalake_env.lower() == "delta":
            table_name = f"{env}_{job_id}_task_log_table"
            output_path = f"{db_name}.{table_name}"
            path = spark.sql(f"desc formatted {output_path}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]
    if table_id:
        h1 = get_headers(vault_scope)
        response = requests.get(
            API_ENDPOINT + TABLES_METADATA, params={"table_id": table_id}, headers=h1
        )
        utils.log(
            f"\n\
        endpoint - {TABLES_ADD}\n\
        response - {response}",
            message_run,
        )
        path = response.json()["data"][0]["dbfs_path"]
    return path

def get_table_db_path(table_type, table_sub_type,catalog_details,table_id=None):
    """
    Returns templated dbfs path as per the table type and sub type
    """
    platform_cloud_provider = catalog_details.get("platform_cloud_provider", "hive")
    if table_type == "Monitoring_Output":
        if table_sub_type == "Data_Model":
            if platform_cloud_provider == "databricks_uc":
                catalog_name = catalog_details.get("catalog_name", "")
                table_name = f"{env}_monitoring_{job_id}_data_model_drift"
                path = f"{catalog_name}.{db_name}.{table_name}"
            elif datalake_env.lower() == "delta":
                table_name = f"{env}_monitoring_{job_id}_data_model_drift"
                path = f"{db_name}.{table_name}"
            else:
                path = f"{env}_monitoring_{job_id}_data_model_drift"
        else:
            if platform_cloud_provider == "databricks_uc":
                catalog_name = catalog_details.get("catalog_name", "")
                table_name = f"{env}_monitoring_{job_id}_performance_drift"
                path = f"{catalog_name}.{db_name}.{table_name}"
            elif datalake_env.lower() == "delta":
                table_name = f"{env}_monitoring_{job_id}_performance_drift"
                path = f"{db_name}.{table_name}"
            else:
                path = f"{env}_monitoring_{job_id}_performance_drift"

    elif table_type == "Task_Log":
        if platform_cloud_provider == "databricks_uc":
            catalog_name = catalog_details.get("catalog_name", "")
            table_name = f"{env}_{job_id}_task_log_table"
            path = f"{catalog_name}.{db_name}.{table_name}"
        elif datalake_env.lower() == "delta":
            table_name = f"{env}_{job_id}_task_log_table"
            path = f"{db_name}.{table_name}"
    if table_id:
        h1 = get_headers(vault_scope)
        response = requests.get(
            API_ENDPOINT + TABLES_METADATA, params={"table_id": table_id}, headers=h1
        )
        utils.log(
            f"\n\
        endpoint - {TABLES_ADD}\n\
        response - {response}",
            message_run,
        )
        path = response.json()["data"][0]["db_path"]
    return path

@retry(
    wait=wait_exponential(min=4, multiplier=1, max=10),
    stop=(stop_after_delay(40) | stop_after_attempt(5)),
)
def add_table_in_mongo(table_type, table_sub_type,catalog_details):
    """
    Adds table in mongo as per the table type and sub type
    """
    # Fetching templated dbfs path
    table_path = get_table_dbfs_path(table_type, table_sub_type, catalog_details)
    db_table_path = get_table_db_path(table_type, table_sub_type, catalog_details)

    # local primary keys
    if table_type == "Monitoring_Output":
        local_primary_keys = [
            "deployment_master_id",
            "monitoring_type",
            "monitoring_sub_type",
            "job_id",
            "run_id",
        ]
    else:
        local_primary_keys = ["monitoring_subtype", "job_id", "run_id"]

    table_schema = []
    if table_sub_type == 'Data_Model' :
        table_schema = get_metric_table_schema('data_drift')

    elif table_sub_type == 'Performance' :
        table_schema = get_metric_table_schema('performance_drift')

    # Defining payload
    data = {
        "name": (
            f"{table_sub_type}_drift"
            if table_type == "Monitoring_Output"
            else "Task_Log_Table"
        ),
        "type": table_type,
        "sub_type": table_sub_type,
        "job_id": str(job_id),
        "created_run_id": str(run_id),
        "task_id": str(task_run_id),
        "project_id": str(project_id),
        "version": str(version),
        "created_by_id": str(created_by_id),
        "created_by_name": created_by_name,
        "updated_by_id": str(created_by_id),
        "updated_by_name": created_by_name,
        "deployment_master_id": str(deployment_master_id),
        "primary_keys": local_primary_keys,
        "status": "active",
        "workspace_id": spark.conf.get("spark.databricks.workspaceUrl"),
        "table_column_schemas" : get_table_col_schema(table_schema, local_primary_keys),
        "datalake_env": datalake_env,
    }
    platform_cloud_provider = catalog_details.get("platform_cloud_provider", "hive")
    if (datalake_env == "delta" and platform_cloud_provider != "databricks_uc"):
        data["dbfs_path"] = table_path
        data["db_path"] = db_table_path
    elif platform_cloud_provider == "databricks_uc":
        data["catalog_name"] = uc_catalog_name
        data["db_path"] = db_table_path
    else:
        data["db_path"] = f"{gcp_project_id}.{bq_database_name}.{table_path}"

    print("ADDING FOLLOWING TABLE")
    print(table_sub_type)
    print("=======")
    # Calling tables add API
    tables_add(data)


def throw_exception(e):
    """
    Updates job run and job task as failed upon occurence of an exception
    """
    stagemetrics.end()
    taskmetrics.end()

    stage_Df = stagemetrics.create_stagemetrics_DF("PerfStageMetrics")
    task_Df = taskmetrics.create_taskmetrics_DF("PerfTaskMetrics")

    aggregate_compute_metrics = (
        stagemetrics.aggregate_stagemetrics_DF()
        .select("executorCpuTime", "peakExecutionMemory")
        .collect()[0]
        .asDict()
    )

    aggregate_compute_metrics["executorCpuTime"] = (
        aggregate_compute_metrics["executorCpuTime"] / 1000
        if aggregate_compute_metrics["executorCpuTime"]
        else 0
    )
    aggregate_compute_metrics["peakExecutionMemory"] = (
        aggregate_compute_metrics["peakExecutionMemory"] / (1024 * 1024)
        if aggregate_compute_metrics["peakExecutionMemory"]
        else 0
    )

    compute_metrics = {
        "stagemetrics": stage_Df.rdd.map(lambda row: row.asDict()).collect(),
        "taskmetrics": task_Df.rdd.map(lambda row: row.asDict()).collect(),
    }

    utils.log("exception", message_run)
    utils.log(str(e), message_run)
    ts = str(int(time.time() * 1000000))
    job_id, run_id, task_run_id, taskKey = utils.get_job_details(dbutils)
    run_notebook_url = generate_run_notebook_url(job_id, run_id)
    model_inference_info = dbutils.widgets.get("model_inference_info")
    if isinstance(model_inference_info, str):
        model_inference_info = json.loads(model_inference_info)
    project_id, version = (
        model_inference_info["project_id"],
        model_inference_info["version"],
    )
    created_by_id, created_by_name = (
        model_inference_info["created_by_id"],
        model_inference_info["created_by_name"],
    )

    # Updating job run
    message_run.append(
        {
            "time": ts,
            "message": "Monitor Job with job id: "
            + str(job_id)
            + " and run id: "
            + str(run_id)
            + " failed with exception- "
            + str(e)
            + ".",
        }
    )
    message_task.append(
        {
            "time": ts,
            "message": "task0"
            + " within job id: "
            + str(job_id)
            + " and run id: "
            + str(run_id)
            + " failed with exception - "
            + str(e)
            + ".",
        }
    )
    log_data = {
        "project_id": project_id,
        "version": str(version),
        "job_id": str(job_id),
        "run_id": str(run_id),
        "task_id": str(task_run_id),
        "end_time": str(ts),
        "status": "failed",
        "message": message_run,
        "updated_by_id": created_by_id,
        "updated_by_name": created_by_name,
        "job_type": "Monitor",
        "cpu": str(aggregate_compute_metrics.get("executorCpuTime", "NA")),
        "ram": str(aggregate_compute_metrics.get("peakExecutionMemory", "NA")),
        "cluster_info": get_cluster_info(spark),
        "compute_metrics": compute_metrics,
        "run_notebook_url": run_notebook_url,
        "deployment_master_id": deployment_master_id,
    }
    # Log the payload excluding large fields
    log_payload = {
        key: value
        for key, value in log_data.items()
        if key not in ["compute_metrics", "cluster_info", "message"]
    }

    print(f"API URL : {JOB_RUNS_UPDATE}. Payload : {log_payload}")
    response = requests.put(API_ENDPOINT + JOB_RUNS_UPDATE, json=log_data, headers=h1)
    print(
        f"\n\
    Logging task:\n\
    endpoint - {JOB_RUNS_UPDATE}\n\
    status   - {response}\n\
    response - {response.text}"
    )

    utils.save_sparkmeasure_aggregated_tables(
        project_name=project_name,
        job_run_update_payload=log_data,
        compute_usage_metrics=aggregate_compute_metrics,
        stagemetrics=stagemetrics,
        taskmetrics=taskmetrics,
        api_endpoint=API_ENDPOINT,
        headers=h1,
        spark=spark,
        dbutils=dbutils,
    )

    # Updating job task
    task_log_data = {
        "job_id": str(job_id),
        "run_id": str(run_id),
        "task_id": str(task_run_id),
        "end_time": str(ts),
        "status": "failed",
        "message": message_task,
        "updated_by_id": created_by_id,
        "updated_by_name": created_by_name,
        "job_type": "Monitor",
        "run_notebook_url": run_notebook_url,
    }
    response = requests.put(
        API_ENDPOINT + JOB_TASK_UPDATE, json=task_log_data, headers=h1
    )

    print(
        f"\n\
    Logging task:\n\
    endpoint - {JOB_TASK_UPDATE}\n\
    status   - {response}\n\
    response - {response.text}\n\
    payload  - {task_log_data}\n"
    )

    cancel_job_run_data = {"run_id": str(run_id), "status": "failed"}
    response = requests.post(
        API_ENDPOINT + CANCEL_JOB_RUN, json=cancel_job_run_data, headers=h1
    )

    print(
        f"\n\
    Logging task:\n\
    endpoint - {CANCEL_JOB_RUN}\n\
    status   - {response}\n\
    response - {response.text}\n\
    payload  - {cancel_job_run_data}\n"
    )

    # Exiting the job
    dbutils.notebook.exit(e)


# def register_delta_as_hive(db_name, table_name, dbfs_path, spark):
#     spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name};")
#     spark.sql(f"USE {db_name};")
#     spark.sql(
#         f"""
#         CREATE EXTERNAL TABLE IF NOT EXISTS {table_name}
#         USING DELTA
#         LOCATION '{dbfs_path}';
#     """
#     )


@retry(
    wait=wait_exponential(min=4, multiplier=1, max=10),
    stop=(stop_after_delay(10) | stop_after_attempt(5)),
)
def get_model_details(model_artifact_id, h1):
    model_details_response = requests.get(
        f"{API_ENDPOINT}{GET_MODEL_ARTIFACT_API}/get?model_artifact_id={model_artifact_id}",
        headers=h1,
    )
    utils.log(
        f"\n\
    endpoint - {GET_MODEL_ARTIFACT_API}/get?model_artifact_id={model_artifact_id}\n\
    response - {model_details_response}",
        message_run,
    )
    if model_details_response.status_code not in [200, 201]:
        raise Exception(
            f"The {GET_MODEL_ARTIFACT_API} returned :{model_details_response.status_code} response."
        )

    model_details_json = model_details_response.json()
    utils.log(f"MODEL DETAILS: {model_details_json}", message_run)
    return model_details_json["data"][0]


def read_data(data_path, table_id=None,is_platform_table=False):
    env_to_read = datalake_env if not is_platform_table else platform_datalake_env
    if env_to_read == "delta" and table_id is not None:
        return uc_utils.read_data(
                spark=spark,
                sql=sql,
                dbutils=dbutils,
                vault_scope=vault_scope,
                api_endpoint=API_ENDPOINT,
                headers=h1,
                table_id=table_id
            )
    elif env_to_read == "delta":
        return utils.df_read(
            data_path=data_path, spark=spark, resource_type=env_to_read
        )
    else:
        return utils.df_read(
            spark=spark,
            data_path=data_path.split(".")[-1],
            bq_database_name=bq_database_name,
            bq_project_id=gcp_project_id,
            encrypted_service_account=encrypted_sa_details,
            encryption_key=encryption_key,
            resource_type=env_to_read,
        )

def get_catalog_details(deployment_env):

    response = requests.get(
        API_ENDPOINT + GET_CATALOG + deployment_env, headers=h1
    )

    utils.log(
        f"\n\
        Logging task:\n\
        endpoint - {GET_CATALOG}\n\
        status   - {response}\n\
        response - {response.text}",
        message_run,
        "api_info"
    )

    catalog_details = response.json().get("data")
    if isinstance(catalog_details, list):
        catalog_details = catalog_details[0]

    return catalog_details

def write_data(
    data_path,
    dataframe,
    mode,
    db_name,
    table_name,
    partition_by,
    is_platform_table=False,
    primary_keys=[],
):
    try:
        deployment_env = dbutils.widgets.get("deployment_env")
    except:
        deployment_env = "dev"
    env_to_write = datalake_env if not is_platform_table else platform_datalake_env
    if env_to_write == "delta":
        uc_utils.write_data_in_delta(
            spark=spark,
            catalog_details=catalog_details,
            dataframe=dataframe,
            db_name=db_name,
            table_name=table_name,
            mode=mode,
            partition_by=partition_by,
            primary_key=primary_keys,
        )
    else:
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


def table_already_created(table_name, schema_name, catalog_name=""):

    try:
        if catalog_name:
            schema_name = f"{catalog_name}.{schema_name}"

        utils.log(f"schema_name: {schema_name}", message_run)
        tables = spark.catalog.listTables(schema_name)
        for table in tables:
            if table.name == table_name:
                utils.log(f"table: {table_name}, \n table_exists : {True}",message_run)
                return True
        utils.log(f"table: {table_name}, \n table_exists : {False}",message_run)
        return False
    except Exception as e:
        utils.log(f"Error while checking if table_exist for table_name: {table_name}: {e}",message_run)
        return False


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
    elif isinstance(obj, pd.core.series.Series):
        obj = obj.to_list()
        result = find_integer(obj)
        if result is not None:  # If an integer is found, return it
            return result
    elif isinstance(obj, int):  # If the input is an integer
        return obj  # Return the integer
    return None  # If no integer is found, return None


# Handling a JSON string input
def extract_integer(input):
    if isinstance(input, str):  # Check if the input is a string
        try:
            input = json.loads(input)  # Try to parse the JSON string
        except json.JSONDecodeError:
            return None  # Return None if the string cannot be parsed
    return find_integer(input)  # Use the find_integer function to extract the integer


# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitoring Sub-Type based execution Methods

# COMMAND ----------


def update_task_log_as_per_inference_table(monitoring_subtype, required_entry):
    """
    Updates the task log table for monitoring subtype
    """
    # Loading and filtering task log for the given monitoring subtype
    # task_log = (
    #             spark.read.load(task_log_path)
    #             .filter(F.col('monitoring_subtype') == monitoring_subtype)
    #             .orderBy('date', 'timestamp', ascending=False)
    # )

    # # Loading inference task log
    # inference_task_log_path = f"dbfs:/mnt/{az_container_name}/{env}/{project_id}/{version}/{model_inference_info['parent_model_infer_job_id']}/task_log_table"
    # df_inf_task = spark.read.load(inference_task_log_path)

    # # Determination of start marker and end marker
    # df_inf_task = (
    #                 df_inf_task
    #                 .filter(F.col("start_marker") > required_entry['end_marker'])
    #                 .orderBy('date', 'timestamp', ascending=True)
    #             )
    # if df_inf_task.first() != None:
    #     start_marker = df_inf_task.first()['start_marker']
    #     end_marker = df_inf_task.first()['end_marker']
    # else:
    if isinstance(required_entry, pyspark.sql.dataframe.DataFrame):
        required_entry = required_entry.toPandas()
    start_marker = extract_integer(required_entry["start_marker"])
    end_marker = extract_integer(required_entry["end_marker"])
    utils.log(f"start_marker : {start_marker}", message_run)
    utils.log(f"end_marker : {end_marker}", message_run)
    # Determination of table name on which markers have been calculated
    table_name = "inference_table"

    # Updating task log with new metadata
    ts = int(time.time() * 1000000)
    schema = StructType(
        [
            StructField("monitoring_subtype", StringType(), True),
            StructField("start_marker", IntegerType(), True),
            StructField("end_marker", IntegerType(), True),
            StructField("table_name", StringType(), True),
        ]
    )
    df_column_name = ["monitoring_subtype", "start_marker", "end_marker", "table_name"]
    df_record = [(monitoring_subtype, int(start_marker), int(end_marker), table_name)]
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
        try:
            df_task = df_task.withColumn("inference_id", F.lit(required_entry["id"]))
        except:
            pass

    # Adding id column
    try:
        existing_df_task = read_data(task_log_path, is_platform_table=True)
        max_id = int(existing_df_task.select(F.max("id")).first()[0])
    except:
        max_id = 1

    try:
        df_task = df_task.withColumn("id", F.lit(max_id + 1))
        write_data(
            data_path=task_log_path,
            dataframe=df_task,
            mode="append",
            db_name = db_name,
            table_name = task_log_path,
            partition_by=None,
            is_platform_table=True,
        )
    except:
        if "id" in df_task.columns:
            df_task = df_task.drop("id")
        write_data(
            data_path=task_log_path,
            dataframe=df_task,
            mode="append",
            table_name = task_log_path,
            db_name = db_name,
            partition_by=None,
            is_platform_table=True,
        )

    print(df_task.display())


# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitoring Specific Helper Methods

# COMMAND ----------


# Function to get numeric columns
def list_numerical_columns(data):
    schema = data.dtypes
    numerical_cols = [
        x[0] for x in schema if x[1] not in ["string", "date", "boolean", "timestamp"]
    ]
    return numerical_cols


def list_datelike_columns(data):
    schema = data.dtypes
    date_cols = [x[0] for x in schema if x[1] in ["date", "timestamp"]]
    return date_cols


# Algo ID to name mapper
def map_algo_id_2_name(algo_id_list):
    algo_name_list = []
    for id in algo_id_list:
        if id == "DD1":
            name = "lsdd"
        elif id == "DD2":
            name = "mmd"
        elif id == "DD3":
            name = "cammd"
        elif id == "DD4":
            name = "lkmmd"
        elif id == "DD5":
            name = "spotdiff"
        elif id == "DD6":
            name = "ks"
        elif id == "DD7":
            name = "emperical_mmd"
        elif id == "DD8":
            name = "chisquare"
        elif id == "DD9":
            name = "fet_ev"
        elif id == "DD10":
            name = "cvm_ev"
        elif id == "FD1":
            name = "mtd"
        elif id == "FD2":
            name = "psi"
        elif id == "FD3":
            name = "acf"
        elif id == "FD4":
            name = "additive-ts-decompose"
        elif id == "FD5":
            name = "pacf"
        elif id == "FD6":
            name = "js"
        elif id == "FD7":
            name = "anderson"
        elif id == "FD8":
            name = "psi_ev"
        elif id == "FD9":
            name = "hellinger"
        elif id == "FD10":
            name = "mann_witney"
        elif id == "FD11":
            name = "ks"
        elif id == "FD12":
            name = "kl_div"
        elif id == "FD13":
            name = "chisquare"
        elif id == "FD14":
            name = "multiplicative-ts-decompose"
        elif id == "FD15":
            name = "z_test"
        elif id == "FD16":
            name = "wasserstein"
        elif id == "FD17":
            name = "fet_ev"
        elif id == "FD18":
            name = "cvm_ev"
        elif id == "FD19":
            name = "g_test"
        elif id == "FD20":
            name = "energy_distance"
        elif id == "FD21":
            name = "epps_singleton"
        elif id == "FD22":
            name = "t_test"
        elif id == "FD23":
            name = "emperical_mmd"
        elif id == "FD24":
            name = "tvd"
        elif id == "TD1":
            name = "lsdd"
        elif id == "TD2":
            name = "mmd"
        elif id == "TD3":
            name = "mtd"
        elif id == "TD4":
            name = "js"
        elif id == "TD5":
            name = "anderson"
        elif id == "TD6":
            name = "psi_ev"
        elif id == "TD7":
            name = "hellinger"
        elif id == "TD8":
            name = "mann_witney"
        elif id == "TD9":
            name = "ks"
        elif id == "TD10":
            name = "kl_div"
        elif id == "TD11":
            name = "chisquare"
        elif id == "TD12":
            name = "additive-ts-decompose"
        elif id == "TD13":
            name = "multiplicative-ts-decompose"
        elif id == "TD14":
            name = "acf"
        elif id == "TD15":
            name = "pacf"
        elif id == "TD16":
            name = "z_test"
        elif id == "TD17":
            name = "wasserstein"
        elif id == "TD18":
            name = "fet_ev"
        elif id == "TD19":
            name = "cvm_ev"
        elif id == "TD20":
            name = "g_test"
        elif id == "TD21":
            name = "energy_distance"
        elif id == "TD22":
            name = "epps_singleton"
        elif id == "TD23":
            name = "t_test"
        elif id == "TD24":
            name = "emperical_mmd"
        elif id == "TD25":
            name = "tvd"
        elif id == "CD1":
            name = "fet"
        elif id == "CD2":
            name = "cvm"
        elif id == "CD3":
            name = "fet_ev"
        elif id == "CD4":
            name = "cvm_ev"
        elif id == "r2":
            name = "r2"
        elif id == "mse":
            name = "mse"
        elif id == "mae":
            name = "mae"
        elif id == "rmse":
            name = "rmse"
        elif id == "accuracy":
            name = "accuracy"
        elif id == "f1":
            name = "f1"
        elif id == "log_loss":
            name = "log_loss"
        elif id == "precision":
            name = "precision"
        elif id == "recall":
            name = "recall"
        elif id == "roc_auc":
            name = "roc_auc"
        elif id == "smape":
            name = "smape"
        elif id == "mdape":
            name = "mdape"
        elif id == "directional_accuracy":
            name = "mda"
        elif id == "mape":
            name = "mape"
        else:
            name = None
        algo_name_list.append(name)
    return algo_name_list


# ModelType ID to name mapper
def map_modeltype_id_2_name(model_id):
    if model_id == "Regression":
        return "regression"
    if model_id == "Classification":
        return "classification"
    if model_id == "Forecasting":
        return "forecasting"
    if model_id == "MMM":
        return "mmm"


# Function to round monitoring metrics
def round_metrics(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = round_metrics(data[key])
    elif isinstance(data, (list, tuple)):
        for i, val in enumerate(data):
            data[i] = round_metrics(val)
    elif isinstance(data, (int, float)):
        data = round(data, 5)
    return data


# Function to handle nan and None in JSON so that the JSON is complaint
def json_encode_nans(data: dict):
    """Custom Encoders are needed when json.dumps doesnt know how to handle some dtypes.
    But on this case np.nan and Nones are something json.dumps knows how to handle.
    So we cannot handle these tupes in custom JSON encoders.
    To override nans and Nones, this recursive function is written"""

    if isinstance(data, dict):
        for key in data.keys():
            data[key] = json_encode_nans(data[key])
    elif isinstance(data, (list, tuple)):
        for i, val in enumerate(data):
            data[i] = json_encode_nans(val)
    elif (str(data) in ["nan", "inf", "-inf"]) or (data is None):
        data = ""
    return data


# Function to check for monitor setup
def check_for_monitor_setup(drift_conf, target_type=None):
    monitor_algo_to_setup = []
    listed_monitors = []
    master_algo_list = []

    if drift_conf["monitoring_type"] == "feature_drift":
        feature_algo_list = drift_conf["monitoring_algo_features"]
        master_algo_list.append(feature_algo_list)
        overall_algo_list = drift_conf["monitoring_algo_overall"]
        master_algo_list.append(overall_algo_list)
        monitoring_type = drift_conf["monitoring_type"]
    elif drift_conf["monitoring_type"] == "target_drift":
        target_algo_list = drift_conf["monitoring_algo"]
        master_algo_list.append(target_algo_list)
        monitoring_type = drift_conf["monitoring_type"] + "_" + target_type
    elif drift_conf["monitoring_type"] == "concept_drift":
        concept_algo_list = drift_conf["monitoring_algo"]
        master_algo_list.append(concept_algo_list)
        monitoring_type = drift_conf["monitoring_type"]
    elif drift_conf["monitoring_type"] == "performance_drift":
        performance_algo_list = drift_conf["monitoring_metrics"]
        master_algo_list.append(performance_algo_list)
        monitoring_type = drift_conf["monitoring_type"]
    try:
        for algo_list in master_algo_list:
            for algo in algo_list:
                algo_name = map_algo_id_2_name([algo["algo_id"]])[0]
                monitoring_artifact_path_for_algo = generate_monitoring_artifact_path(
                    monitoring_type=monitoring_type,
                    monitoring_artifacts_base_dir=monitoring_artifact_path,
                    monitor_algo=algo_name,
                )
                listed_monitors.append(algo_name)
                utils.log(f"{monitoring_artifact_path_for_algo}", message_run)
                if os.path.exists(monitoring_artifact_path_for_algo) and os.path.isdir(
                    monitoring_artifact_path_for_algo
                ):
                    # Artifacts are already generated, proceed to monitor
                    pass
                else:
                    # Add to list of algorithms to setup
                    monitor_algo_to_setup.append(algo_name)

        utils.log(f"listed_feature_monitors : {listed_monitors}", message_run)
        utils.log(
            f"feature_monitor_algo_to_setup : {monitor_algo_to_setup}", message_run
        )
    except Exception as e:
        traceback.print_exc()

    return monitor_algo_to_setup, listed_monitors


# Fuction to setup monitor
def setup_monitor(
    monitor_algo_to_setup,
    drift_conf,
    data_type,
    monitoring_artifact_path,
    reference_df=None,
    target=None,
    target_type=None,
    base_y_actuals=None,
    base_y_predicted=None,
    date_column=None,
):
    artifact_path = "No Artifacts Created"

    # Sample size selection
    DATA_DRIFT_SAMPLE = 1000
    MODEL_DRIFT_SAMPLE = 1000

    if not monitor_algo_to_setup:
        utils.log(f"artifact_path : {artifact_path}", message_run)
        return
    try:
        if drift_conf["monitoring_type"] == "feature_drift":
            if reference_df.shape[0] < 1000:
                DATA_DRIFT_SAMPLE = reference_df.shape[0]
            artifact_path = setup_feature_drift_monitors(
                data_type=data_type,
                reference_df=reference_df.sample(n=DATA_DRIFT_SAMPLE),
                monitoring_artifacts_base_dir=monitoring_artifact_path,
                monitored_features=drift_conf["monitored_features"],
                monitoring_algos=monitor_algo_to_setup,
            )
        elif drift_conf["monitoring_type"] == "target_drift":
            if drift_conf["monitoring_mode"] == "Batch":
                if reference_df.shape[0] < 1000:
                    DATA_DRIFT_SAMPLE = reference_df.shape[0]
                artifact_path = setup_target_drift_monitors(
                    data_type=data_type,
                    reference_df=reference_df.sample(n=DATA_DRIFT_SAMPLE),
                    target=target,
                    target_type=target_type,
                    monitoring_artifacts_base_dir=monitoring_artifact_path,
                    monitoring_algos=monitor_algo_to_setup,
                    date_column=date_column,
                )
        elif drift_conf["monitoring_type"] == "concept_drift":
            if drift_conf["monitoring_mode"] == "Batch":
                if base_y_actuals.shape[0] < 1000:
                    MODEL_DRIFT_SAMPLE = base_y_actuals.shape[0]
                artifact_path = setup_concept_drift_monitors(
                    data_type=data_type,
                    model_task=map_modeltype_id_2_name(
                        drift_conf["modelling_task_type"]
                    ),
                    base_y_actuals=base_y_actuals[:MODEL_DRIFT_SAMPLE],
                    base_y_predicted=base_y_predicted[:MODEL_DRIFT_SAMPLE],
                    monitoring_algos=monitor_algo_to_setup,
                    monitoring_artifacts_base_dir=monitoring_artifact_path,
                )
        elif drift_conf["monitoring_type"] == "performance_drift":
            if drift_conf["monitoring_mode"] == "Batch":
                artifact_path = setup_performance_drift_monitors(
                    data_type=data_type,
                    model_task=map_modeltype_id_2_name(
                        drift_conf["modelling_task_type"]
                    ),
                    base_y_actuals=base_y_actuals,
                    base_y_predicted=base_y_predicted,
                    monitoring_metrics=monitor_algo_to_setup,
                    monitoring_artifacts_base_dir=monitoring_artifact_path,
                )
        utils.log(f"artifact_path : {artifact_path}", message_run)
    except Exception as e:
        traceback.print_exc()


# Function to monitor current data
def monitor_current_data(
    drift_conf,
    data_type,
    monitoring_artifact_path,
    monitoring_algos,
    current_df=None,
    target=None,
    target_type=None,
    current_y_actuals=None,
    current_y_predicted=None,
    date_column=None,
):
    drift_metrics_list = []
    try:
        if (current_df is not None) and (current_df.shape[0] == 0):
            raise Exception("There is no current data to monitor")
        elif (current_y_actuals is not None) and (current_y_actuals.shape[0] == 0):
            raise Exception("There is no current data to monitor")
        elif (current_y_predicted is not None) and (current_y_predicted.shape[0] == 0):
            raise Exception("There is no current data to monitor")

        # Sample size selection
        DATA_DRIFT_SAMPLE = 1000
        MODEL_DRIFT_SAMPLE = 1000

        if drift_conf["monitoring_type"] == "feature_drift":
            if drift_conf["monitoring_mode"] == "Batch":
                if current_df.shape[0] < 1000:
                    DATA_DRIFT_SAMPLE = current_df.shape[0]
                drift_metrics_list = monitor_feature_drift(
                    data_type=data_type,
                    current_df=current_df.sample(n=DATA_DRIFT_SAMPLE),
                    monitoring_artifacts_base_dir=monitoring_artifact_path,
                    monitored_features=drift_conf["monitored_features"],
                    monitoring_algos=monitoring_algos,
                    monitoring_algo_thresholds=drift_conf[
                        "monitoring_algo_feature_thresholds"
                    ]
                    + drift_conf["monitoring_algo_overall_thresholds"],
                )
        elif drift_conf["monitoring_type"] == "target_drift":
            if drift_conf["monitoring_mode"] == "Batch":
                if current_df.shape[0] < 1000:
                    DATA_DRIFT_SAMPLE = current_df.shape[0]
                drift_metrics_list = monitor_target_drift(
                    data_type=data_type,
                    current_df=current_df.sample(n=DATA_DRIFT_SAMPLE),
                    target=target,
                    target_type=target_type,
                    monitoring_artifacts_base_dir=monitoring_artifact_path,
                    monitoring_algos=monitoring_algos,
                    monitoring_algo_thresholds=drift_conf["monitoring_algo_thresholds"],
                    date_column=date_column,
                )
        elif drift_conf["monitoring_type"] == "concept_drift":
            if drift_conf["monitoring_mode"] == "Batch":
                if current_y_actuals.shape[0] < 1000:
                    MODEL_DRIFT_SAMPLE = current_y_actuals.shape[0]
                drift_metrics_list = monitor_concept_drift(
                    data_type=data_type,
                    current_y_actuals=current_y_actuals[:MODEL_DRIFT_SAMPLE],
                    current_y_predicted=current_y_predicted[:MODEL_DRIFT_SAMPLE],
                    monitoring_algos=monitoring_algos,
                    monitoring_artifacts_base_dir=monitoring_artifact_path,
                    monitoring_algo_thresholds=drift_conf["monitoring_algo_thresholds"],
                )
                for metric in drift_metrics_list:
                    metric["stat_val"] = metric["stat_val"][0]
                    metric["p_val"] = metric["p_val"][0]
        elif drift_conf["monitoring_type"] == "performance_drift":
            if drift_conf["monitoring_mode"] == "Batch":
                drift_json = monitor_performance_drift(
                    data_type=data_type,
                    model_task=map_modeltype_id_2_name(
                        drift_conf["modelling_task_type"]
                    ),
                    current_y_actuals=current_y_actuals,
                    current_y_predicted=current_y_predicted,
                    monitoring_metrics=monitoring_algos,
                    monitoring_artifacts_base_dir=monitoring_artifact_path,
                    monitoring_metrics_thresholds=drift_conf[
                        "monitoring_algo_thresholds"
                    ][0],
                )
                drift_metrics_list = [drift_json]
    except Exception as e:
        traceback.print_exc()

    return drift_metrics_list


def delta_encode_str(data: dict):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = delta_encode_str(data[key])
    elif isinstance(data, (list, tuple)):
        for i, val in enumerate(data):
            data[i] = delta_encode_str(val)
    elif str(data) in [" ", "", "", " "]:
        data = None
    return data


# COMMAND ----------

# MAGIC %md
# MAGIC ## Metrics write into delta Helper Methods

# COMMAND ----------

def get_table_col_schema(df_schema, primary_keys):
    """
    Returns dataframe schema to be updated in Tables collection
    """
    # Mapping dictionary
    type_mapping = {
        StringType: "string",
        IntegerType: "bigint",
        DoubleType: "double",
        BooleanType: "bool",
        LongType: "bigint",
        DateType : 'date',
        MapType : 'map<string,string>'
    }
    try:
        data_types = [(field.name, type_mapping[type(field.dataType)]) for field in df_schema]
        df_datatypes = dict(data_types)
        return [
                    {
                        'column_name': str(entry.name),
                        'datatype': df_datatypes[str(entry.name)],
                        'nullable': str(entry.nullable),
                        'is_primary': "1" if str(entry.name) in primary_keys else '0'
                    } for entry in df_schema
               ]
    except Exception as e:
        return []

def get_metric_table_schema(kind):
    """
    Returns the predefined schema for the dedicated table for the kind of monitoring metrics to be written
    """
    if kind != "performance_drift":
        data_model_schema = StructType(
            [
                StructField("feature_attribute", StringType(), True),
                StructField("is_drift", LongType(), True),
                StructField("stat_val", DoubleType(), True),
                StructField("p_val", DoubleType(), True),
                StructField("model_artifact_id", StringType(), True),
                StructField("input_table_id", StringType(), True),
                StructField("data_prep_deployment_job_id", StringType(), True),
                StructField("deployment_master_id", StringType(), True),
                StructField("processing_time", LongType(), True),
                StructField("batch_start_id", StringType(), True),
                StructField("batch_end_id", StringType(), True),
                StructField("batch_size", LongType(), True),
                StructField("monitoring_type", StringType(), True),
                StructField("monitoring_sub_type", StringType(), True),
                StructField("env", StringType(), True),
                StructField("created_by_name", StringType(), True),
                StructField("created_by_id", StringType(), True),
                StructField(
                    "batch_markers", MapType(StringType(), StringType(), True), True
                ),
                StructField("monitoring_algo_id", StringType(), True),
                StructField("monitoring_algo_name", StringType(), True),
                StructField("job_id", StringType(), True),
                StructField("run_id", StringType(), True),
                StructField("modelling_task_name", StringType(), True),
                StructField("model_name", StringType(), True),
                StructField("model_version", StringType(), True),
                StructField("model_session_id", StringType(), True),
                StructField("project_id", StringType(), True),
                StructField("project_name", StringType(), True),
                StructField("version", StringType(), True),
                StructField("inference_metadata", StringType(), True),
            ]
        )
        return data_model_schema

    else:
        performance_data_schema = StructType(
            [
                StructField("feature_attribute", StringType(), True),
                StructField("is_drift", LongType(), True),
                StructField("value", DoubleType(), True),
                StructField("model_artifact_id", StringType(), True),
                StructField("input_table_id", StringType(), True),
                StructField("data_prep_deployment_job_id", StringType(), True),
                StructField("deployment_master_id", StringType(), True),
                StructField("processing_time", LongType(), True),
                StructField("batch_start_id", StringType(), True),
                StructField("batch_end_id", StringType(), True),
                StructField("batch_size", LongType(), True),
                StructField("monitoring_type", StringType(), True),
                StructField("monitoring_sub_type", StringType(), True),
                StructField("env", StringType(), True),
                StructField("created_by_name", StringType(), True),
                StructField("created_by_id", StringType(), True),
                StructField(
                    "batch_markers", MapType(StringType(), StringType(), True), True
                ),
                StructField("modelling_task_name", StringType(), True),
                StructField("monitoring_algo_name", StringType(), True),
                StructField("job_id", StringType(), True),
                StructField("run_id", StringType(), True),
                StructField("model_name", StringType(), True),
                StructField("model_version", StringType(), True),
                StructField("model_session_id", StringType(), True),
                StructField("project_id", StringType(), True),
                StructField("project_name", StringType(), True),
                StructField("version", StringType(), True),
                StructField("inference_metadata", StringType(), True),
            ]
        )
        return performance_data_schema


# COMMAND ----------

# MAGIC %md
# MAGIC ## Global Variables

# COMMAND ----------

message_run = []
message_task = []

# Fetching env and scope information
env, vault_scope = get_env_vault_scope()
secrets_object = fetch_secrets_from_dbutils(dbutils, message_run)
az_container_name = secrets_object.get("az-container-name","")

# API Endpoints
try:
    API_ENDPOINT = dbutils.widgets.get("tracking_base_url")
except:
    API_ENDPOINT = get_app_url()
SCHEDULE_STATUS = "mlapi/task/schedule/status"
JOB_TASK_ADD = "mlapi/job/task/log/add"
TABLES_ADD = "mlapi/tables/add"
JOB_TASK_UPDATE = "mlapi/job/task/log/update"
JOB_RUNS_UPDATE = "mlapi/job/runs/log/update"
TRANSFORMS_SCHEMA = "mlapi/transforms/schema/list"
EDA_RUN = "mlapi/eda/run"
LIST_TABLES = "mlapi/tables/list"
TABLES_UPDATE = "mlapi/tables/update"
CANCEL_JOB_RUN = "mlapi/jobs/run/cancel"
TASKS_STATUS = "mlapi/job/run/tasks/status"
RUN_TASK_ADD = "mlapi/job/runs/log/add"
JOB_LOGS = "mlapi/jobs_logs/list"
GET_MODEL_ARTIFACT_API = "mlapi/modelartifact"
TABLES_METADATA = "mlapi/tables/metadata/list"
GET_CATALOG = "mlapi/get_catalog?deployment_env="

# Text String to get the API
h1 = get_headers(vault_scope)

# Notebook Params
job_id, run_id, task_run_id, taskKey = utils.get_job_details(dbutils)
run_notebook_url = generate_run_notebook_url(job_id, run_id)
model_inference_info = dbutils.widgets.get("model_inference_info")
if isinstance(model_inference_info, str):
    model_inference_info = json.loads(model_inference_info)

project_id, version = (
    model_inference_info["project_id"],
    model_inference_info["version"],
)
created_by_id, created_by_name = (
    model_inference_info["created_by_id"],
    model_inference_info["created_by_name"],
)
model_artifact_id = model_inference_info["model_artifact_id"]
model_name = model_inference_info["model_name"]
# Reference and Current table metadata
reference_table_path = model_inference_info["reference_table"]["path"]
current_table_path = model_inference_info["current_table"]["path"]
reference_table_id = model_inference_info["reference_table"]["table_id"]
current_table_id = model_inference_info["current_table"]["table_id"]
inference_table_path = model_inference_info["inference_output_table_path"]
yactuals_table_path = model_inference_info["y_actual_table_path"]
inference_table_id = model_inference_info["inference_output_table_id"]
yactuals_table_id = model_inference_info["y_actual_table_id"]
deployment_master_id = model_inference_info["deployment_master_id"]
db_name = f"{project_id}_{version}".lower()
project_name = model_inference_info.get("project_name", "")


# deployment env
now = datetime.now()
date = now.strftime("%m-%d-%Y")

try:
    batch_size = int(get_params("batch_size"))
    if batch_size == 0:
        batch_size = 10000
except:
    batch_size = 10000

try:
    env = dbutils.widgets.get("env")
except:
    env = "dev"

try:
    deployment_env = dbutils.widgets.get("deployment_env")
except:
    deployment_env = "dev"

try:
    datalake_env = dbutils.widgets.get("datalake_env")
except Exception as e:
    utils.log(f"Exception while retrieving data lake environment : {e}", message_run)
    datalake_env = "delta"

primary_keys = model_inference_info.get("primary_keys", [])
if primary_keys in ["", None]:
    primary_keys = []
if isinstance(primary_keys, str):
    primary_keys = json.loads(primary_keys)

target_columns = model_inference_info.get("target_columns", [])
if target_columns in ["", None]:
    target_columns = []
if isinstance(target_columns, str):
    try:
        target_columns = json.loads(target_columns)
    except:
        try:
            target_columns = json.loads(target_columns.replace("'", '"'))
        except:
            target_columns = ast.literal_eval(target_columns)

catalog_details = get_catalog_details(deployment_env)
platform_datalake_env = secrets_object.get("platform-datalake-env","")
platform_db_name = f"mlcore_observability_{deployment_env}"
platform_cloud_provider = catalog_details.get("platform_cloud_provider", "hive")
uc_catalog_name = (
    catalog_details.get("catalog_name", "")
    if platform_cloud_provider == "databricks_uc"
    else ""
)

# Defining inference task log path
if model_inference_info.get("inference_task_log_table_id", None):
    inference_task_log_path = get_table_dbfs_path(
        "", "", catalog_details,model_inference_info["inference_task_log_table_id"]
    )
else:
    if platform_datalake_env.lower() == "delta":
        inference_task_log_path = f"dbfs:/mnt/{az_container_name}/{env}/{project_id}/{version}/{model_inference_info['parent_model_infer_job_id']}/task_log_table"
    else:
        encrypted_sa_details = secrets_object.get("gcp-service-account-encypted","")
        encryption_key = secrets_object.get("gcp-service-account-private-key","")
        bq_database_name = secrets_object.get("gcp-bq-database-name","")
        gcp_project_id = secrets_object.get("gcp-api-quota-project-id","")

        inference_task_log_path = f"{env}_{project_id}_{version}_{model_inference_info['parent_model_infer_job_id']}_task_log_table"


# COMMAND ----------

# Checking if upstream jobs, i.e. DPD and Model Inference, are active or not, skipping the run if not active.
data_prep_deployment_job_id = model_inference_info.get(
    "data_prep_deployment_job_id", ""
)
model_inference_job_id = model_inference_info.get("parent_model_infer_job_id", "")

job_ids_to_check = []
if data_prep_deployment_job_id != "" and data_prep_deployment_job_id != None:
    job_ids_to_check.append(data_prep_deployment_job_id)

if model_inference_job_id != "" and model_inference_job_id != None:
    job_ids_to_check.append(model_inference_job_id)

if not check_if_upstream_job_active(job_ids_to_check):
    declare_job_as_successful(upstream_inactive=True)

# COMMAND ----------

try:
    # Updating the task log info
    ts = str(int(time.time() * 1000000))
    log_data = {
        "project_id": project_id,
        "version": version,
        "job_id": str(job_id),
        "run_id": str(run_id),
        "tasks_list": [task_run_id],
        "add_task_logs": "yes",
        "status": "running",
        "start_time": str(ts),
        "created_by_id": created_by_id,
        "created_by_name": created_by_name,
        "job_type": "Monitor",
        "project_name": project_name,
        "deployment_master_id": deployment_master_id,
        "run_notebook_url": run_notebook_url,
    }
    response = requests.post(API_ENDPOINT + RUN_TASK_ADD, json=log_data, headers=h1)
    utils.log(
        f"\n\
    Logging task:\n\
    endpoint - {RUN_TASK_ADD}\n\
    status   - {response}\n\
    payload  - {log_data}",
        message_run,
    )

    task_log_data = {
        "project_id": project_id,
        "version": version,
        "job_id": str(job_id),
        "run_id": str(run_id),
        "task_id": str(task_run_id),
        "status": "running",
        "start_time": str(ts),
        "created_by_id": created_by_id,
        "created_by_name": created_by_name,
        "job_type": "Monitor",
    }
    response = requests.post(
        API_ENDPOINT + JOB_TASK_ADD, json=task_log_data, headers=h1
    )
    utils.log(
        f"\n\
    Logging task:\n\
    endpoint - {JOB_TASK_ADD}\n\
    status   - {response}\n\
    response - {response.text}\n\
    payload  - {task_log_data}\n",
        message_run,
    )

    # Updating message run and message task
    t = str(int(time.time() * 1000000))
    message_run.append({"time": t, "message": "Monitor Task has been scheduled."})
    message_task.append({"time": t, "message": "Monitor Task has been scheduled."})

except Exception as e:
    throw_exception(e)

# COMMAND ----------

try:
    # Fetching monitoring config
    monitoring_config = get_monitoring_config(deployment_master_id)
    if monitoring_config == []:
        declare_job_as_successful(no_monitor_config=True)
except Exception as e:
    throw_exception(e)

# COMMAND ----------

try:
    monitoring_activation_info = {
        "feature_drift": {"is_activated": "1", "start_date": "", "end_date": ""},
        "concept_drift": {"is_activated": "1", "start_date": "", "end_date": ""},
        "target_y_pred": {"is_activated": "1", "start_date": "", "end_date": ""},
        "target_y_actual": {"is_activated": "1", "start_date": "", "end_date": ""},
        "performance_drift": {"is_activated": "1", "start_date": "", "end_date": ""},
    }
    if "controls" in monitoring_config.keys():
        if len(monitoring_config["controls"]) > 0:
            monitoring_activation_info = monitoring_config["controls"][
                "monitor_control"
            ]
except Exception as e:
    throw_exception(e)

# COMMAND ----------

# DBTITLE 1,Get Model Artifact Details
model_details_json = get_model_details(model_artifact_id, h1)

model_name = model_details_json["model_name"]
model_version = model_details_json["model_version"]
model_session_id = model_details_json["model_train_session_id"]
modelling_task_type = model_details_json["model_train_variables"]["modelling_task_type"]
feature_columns = model_details_json["feature_columns"]
target_column = model_details_json["target_columns"]

if not isinstance(feature_columns, list):
    feature_columns = ast.literal_eval(feature_columns)
if not isinstance(target_column, list):
    target_column = ast.literal_eval(target_column)

prediction_column = ["prediction"]

try:
    partition_keys = ast.literal_eval(dbutils.widgets.get("partition_keys"))
except:
    partition_keys = []

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get reference table

# COMMAND ----------

if datalake_env.lower() == "delta":

    monitoring_artifact_path = f"dbfs:/mnt/{az_container_name}/{env}/{project_id}/{version}/monitoring/{reference_table_id}/{current_table_id}/{model_artifact_id}"
else:
    monitoring_artifact_path = f"{env}_{project_id}_{version}_monitoring_{reference_table_id}_{current_table_id}_{model_artifact_id}"

if platform_datalake_env == "delta":
    task_log_path = f"{env}_{job_id}_task_log_table"
    # task_log_path = get_table_dbfs_path("Task_Log","Monitor_Batch",catalog_details)
else:
    task_log_path = f"{env}_{project_id}_{version}_{job_id}_task_log_table"

# COMMAND ----------

table_details = {
    "model_name": model_name,
    "table_path": {
        "source_table_path": reference_table_path,
        "ground_truth_table_path": yactuals_table_path,
        "inference_task_log_path": inference_task_log_path,
        "inference_table_path": inference_table_path,
        "monitoring_task_log_path": task_log_path,
        "source_table_id": reference_table_id,
        "ground_truth_table_id": yactuals_table_id,
        "inference_table_id": inference_table_id,
        "inference_task_log_table_id":model_inference_info["inference_task_log_table_id"],
    },
    "feature_columns": feature_columns,
    "gt_column": target_column,
    "prediction_column": prediction_column,
    "primary_keys": primary_keys,
    "partition_keys": partition_keys,
}

utils.log(table_details, message_run)

# COMMAND ----------


reference_table, current_table, inference_task_record = get_monitoring_tables(
    dbutils, spark,  catalog_details, db_name ,table_details
)



# COMMAND ----------

reference_table.display()

# COMMAND ----------

current_table.display()

# COMMAND ----------

print(inference_task_record)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Local Variables

# COMMAND ----------

utils.log(
    f"input params :\n\
    project_id     - {project_id},\n\
    version         - {version},\n\
    created_by_id   - {created_by_id},\n\
    datalake_env    - {datalake_env},\n\
    batch_size  - {batch_size},\n\
    mode -   {env},\n\
    model_artifact_id  - {model_artifact_id},\n\
    model_name      - {model_name},\n\
    model_version   - {model_version},\n\
    model_session_id   - {model_session_id},\n\
    modelling_task_type - {modelling_task_type},\n\
    reference_table_path  - {reference_table_path},\n\
    current_table_path       - {current_table_path},\n\
    task_log_path        - {task_log_path},\n\
    monitoring_artifact_path  - {monitoring_artifact_path},\n\
    reference_table    - {type(reference_table)},\n\
    feature_columns    - {feature_columns},\n\
    ",
    message_run,
)


aggregated_dd_table_name = "aggregated_monitoring_data_drift_table"
aggregated_pd_table_name = "aggregated_monitoring_performance_drift_table"
if platform_datalake_env.lower() == "delta":
    aggregated_performance_drift_path = f"dbfs:/user/hive/warehouse/{platform_db_name}.db/{aggregated_pd_table_name}"
    aggregated_dd_drift_path = f"dbfs:/user/hive/warehouse/{platform_db_name}.db/{aggregated_dd_table_name}"
else:
    aggregated_performance_drift_path = (
        f"{deployment_env}_monitoring_aggregated_performance_drift"
    )
    aggregated_dd_drift_path = f"{deployment_env}_monitoring_aggregated_data_drift"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitoring Business Logic

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Monitoring Imports

# COMMAND ----------

from monitoring.utils.helpers import generate_monitoring_artifact_path

from monitoring.data.feature_drift.batch_monitors import (
    setup_feature_drift_monitors,
    monitor_feature_drift,
    monitor_feature_drift_in_memory,
)
from monitoring.data.target_drift.batch_monitors import (
    setup_target_drift_monitors,
    monitor_target_drift,
)
from monitoring.model.concept_drift.batch_monitors import (
    setup_concept_drift_monitors,
    monitor_concept_drift,
)
from monitoring.model.performance_drift.batch_monitors import (
    setup_performance_drift_monitors,
    monitor_performance_drift,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Monitoring config
# MAGIC

# COMMAND ----------

print(monitoring_config)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get Dataframes for each Monitoring subtype

# COMMAND ----------

# import traceback

# try:
# Fetching required properties
project_id = monitoring_config["project_id"]
version = monitoring_config["version"]
model_artifact_id = monitoring_config["model_artifact_id"]
input_table_id = monitoring_config["input_table_id"]
data_prep_deployment_job_id = monitoring_config["data_prep_deployment_job_id"]
actual_target = "y_actual"
predicted_target = "prediction"
null_gt_skipped_types = []

# Extracting monitoring data as per the configuration
if (
    "dq_drift" in monitoring_config.keys()
    and monitoring_activation_info.get("dq_drift", {}).get("is_activated", "0") == "1"
):
    dq_drift_conf = monitoring_config["dq_drift"]
else:
    dq_drift_conf = {
        "monitoring_type": "dq_drift",
        "monitoring_algo": [],
        "monitoring_mode": None,
    }

if (
    "feature_drift" in monitoring_config.keys()
    and monitoring_activation_info.get("feature_drift", {}).get("is_activated", "0")
    == "1"
):
    # Fetching feature drift configuration
    feature_drift_conf = monitoring_config["feature_drift"]
    reference_df = reference_table.toPandas().dropna(subset=feature_columns)

    # Fetching start date and end date from activation info
    start_date = monitoring_activation_info["concept_drift"]["start_date"]
    end_date = monitoring_activation_info["concept_drift"]["end_date"]

    # Fetching next batch of Inference data
    current_df, current_df_inference_entry = current_table, inference_task_record
    try:
        current_df = current_df.toPandas()
    except:
        utils.log(f"Current Df was sent as None, batches are exhausted.", message_run)
        feature_drift_conf = {
            "monitoring_type": "feature_drift",
            "monitoring_algo_overall": [],
            "monitoring_algo_features": [],
            "monitoring_mode": None,
        }
        reference_df = None
        current_df = None
else:
    utils.log(
        "Either feature_drift key is not there in config OR is_activated is 0",
        message_run,
    )
    feature_drift_conf = {
        "monitoring_type": "feature_drift",
        "monitoring_algo_overall": [],
        "monitoring_algo_features": [],
        "monitoring_mode": None,
    }
    reference_df = None
    current_df = None

if "target_drift" in monitoring_config.keys() and (
    monitoring_activation_info.get("target_y_pred", {}).get("is_activated", "0") == "1"
    or monitoring_activation_info.get("target_y_actual", {}).get("is_activated", "0")
    == "1"
):
    # Fetching target drift configuration
    target_drift_conf = monitoring_config["target_drift"]
    actual_target = "y_actual"
    predicted_target = "prediction"

    # Fetching start date and end date from activation info
    start_date = monitoring_activation_info["concept_drift"]["start_date"]
    end_date = monitoring_activation_info["concept_drift"]["end_date"]

    # Getting y_pred for reference table
    tdp_reference_df = reference_table.toPandas().dropna(subset=["prediction"])

    # Getting inference data joined batch for target y pred
    tdp_current_df, tdp_current_df_inference_entry = (
        current_table,
        inference_task_record,
    )
    try:
        tdp_current_df = tdp_current_df.toPandas()
    except:
        utils.log(
            f"tdp_current_df was sent as None, batches are exhausted.", message_run
        )
        tdp_current_df = None

    # Getting y_true for reference table
    tda_reference_df = reference_table.toPandas().dropna(subset=["y_actual"])

    # Getting inference data joined batch for target y actual
    tda_current_df, tda_current_df_inference_entry = (
        current_table,
        inference_task_record,
    )

    try:
        # If null values exist in actual target column, we skip the execution to maintain inference batch lineage
        y_actual_null_count = tda_current_df.select(
            [
                F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c)
                for c in [actual_target]
            ]
        ).first()[actual_target]

        if y_actual_null_count > 0:
            utils.log(
                "TARGET_DRIFT: If null values exist in actual target column, we skip the execution to maintain inference batch lineage",
                message_run,
            )
            target_drift_conf = {
                "monitoring_type": "target_drift",
                "monitoring_algo": [],
                "monitoring_mode": None,
            }
            tda_reference_df = None
            tda_current_df = None
            tdp_reference_df = None
            tdp_current_df = None
            null_gt_skipped_types.append("target_y_actual")
        else:
            tda_current_df = tda_current_df.toPandas()
    except:
        utils.log("Target Drift Data Fetch Exception", message_run)
        traceback.print_exc()
        target_drift_conf = {
            "monitoring_type": "target_drift",
            "monitoring_algo": [],
            "monitoring_mode": None,
        }
        tda_reference_df = None
        tda_current_df = None
        tdp_reference_df = None
        tdp_current_df = None

else:
    utils.log(
        "Either target_drift key is not there in config OR is_activated is 0",
        message_run,
    )
    target_drift_conf = {
        "monitoring_type": "target_drift",
        "monitoring_algo": [],
        "monitoring_mode": None,
    }
    tda_reference_df = None
    tda_current_df = None
    tdp_reference_df = None
    tdp_current_df = None

if (
    "concept_drift" in monitoring_config.keys()
    and monitoring_activation_info.get("concept_drift", {}).get("is_activated", "0")
    == "1"
):
    # Fetching concept drift configuration
    concept_drift_conf = monitoring_config["concept_drift"]
    actual_target = "y_actual"
    predicted_target = "prediction"

    # Fetching start date and end date from activation info
    start_date = monitoring_activation_info["concept_drift"]["start_date"]
    end_date = monitoring_activation_info["concept_drift"]["end_date"]

    # Getting y_true and y_pred for reference table
    ref_df_w_yact_ypred = (
        reference_table.select([actual_target, predicted_target]).dropna().toPandas()
    )

    # Fetching data required for concept drift
    curr_df_w_yact_ypred_conc = current_table
    curr_df_w_yact_ypred_con_inference_entry = inference_task_record

    try:
        # If null values exist in actual target column, we skip the execution to maintain inference batch lineage
        y_actual_null_count = curr_df_w_yact_ypred_conc.select(
            [
                F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c)
                for c in [actual_target]
            ]
        ).first()[actual_target]
        if y_actual_null_count > 0:
            utils.log(
                "CONCEPT_DRIFT: If null values exist in actual target column, we skip the execution to maintain inference batch lineage",
                message_run,
            )
            concept_drift_conf = {
                "monitoring_type": "concept_drift",
                "monitoring_algo": [],
                "monitoring_mode": None,
            }
            cda_reference_df = None
            cdp_reference_df = None
            cda_current_df = None
            cdp_current_df = None
            null_gt_skipped_types.append("concept_drift")
        else:
            curr_df_w_yact_ypred_conc = curr_df_w_yact_ypred_conc.toPandas()
            cda_reference_df = ref_df_w_yact_ypred[actual_target]
            cdp_reference_df = ref_df_w_yact_ypred[predicted_target]
            cda_current_df = curr_df_w_yact_ypred_conc[actual_target]
            cdp_current_df = curr_df_w_yact_ypred_conc[predicted_target]
    except:
        utils.log("Concept Drift Data Fetch Exception", message_run)
        traceback.print_exc()
        concept_drift_conf = {
            "monitoring_type": "concept_drift",
            "monitoring_algo": [],
            "monitoring_mode": None,
        }
        cda_reference_df = None
        cdp_reference_df = None
        cda_current_df = None
        cdp_current_df = None
else:
    utils.log(
        "Either concept_drift key is not there in config OR is_activated is 0",
        message_run,
    )
    concept_drift_conf = {
        "monitoring_type": "concept_drift",
        "monitoring_algo": [],
        "monitoring_mode": None,
    }
    cda_reference_df = None
    cdp_reference_df = None
    cda_current_df = None
    cdp_current_df = None

if (
    "performance_drift" in monitoring_config.keys()
    and monitoring_activation_info.get("performance_drift", {}).get("is_activated", "0")
    == "1"
):
    # Fetching performance drift configuration
    performance_drift_conf = monitoring_config["performance_drift"]
    for metric_dict in performance_drift_conf["monitoring_metrics"]:
        metric_dict["algo_name"] = metric_dict["metrics"]
    actual_target = "y_actual"
    predicted_target = "prediction"

    # Fetching start date and end date
    start_date = monitoring_activation_info["performance_drift"]["start_date"]
    end_date = monitoring_activation_info["performance_drift"]["end_date"]

    # Getting y_true and y_pred for reference table
    ref_df_w_yact_ypred = (
        reference_table.select([actual_target, predicted_target]).dropna().toPandas()
    )

    # Fetching data required for performance drift
    curr_df_w_yact_ypred_perf = current_table
    curr_df_w_yact_ypred_perf_inference_entry = inference_task_record
    try:
        # If null values exist in actual target column, we skip the execution to maintain inference batch lineage
        y_actual_null_count = curr_df_w_yact_ypred_perf.select(
            [
                F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c)
                for c in [actual_target]
            ]
        ).first()[actual_target]
        if y_actual_null_count > 0:
            utils.log(
                "PERF_DRIFT: If null values exist in actual target column, we skip the execution to maintain inference batch lineage",
                message_run,
            )
            performance_drift_conf = {
                "monitoring_type": "performance_drift",
                "monitoring_metrics": [],
                "monitoring_mode": None,
            }
            pda_reference_df = None
            pdp_reference_df = None
            pda_current_df = None
            pdp_current_df = None
            null_gt_skipped_types.append("performance_drift")
        else:
            curr_df_w_yact_ypred_perf = curr_df_w_yact_ypred_perf.toPandas()
            pda_reference_df = ref_df_w_yact_ypred[actual_target]
            pdp_reference_df = ref_df_w_yact_ypred[predicted_target]
            pda_current_df = curr_df_w_yact_ypred_perf[actual_target]
            pdp_current_df = curr_df_w_yact_ypred_perf[predicted_target]
    except:
        utils.log("Performance drift Data Fetch Exception", message_run)
        traceback.print_exc()
        performance_drift_conf = {
            "monitoring_type": "performance_drift",
            "monitoring_metrics": [],
            "monitoring_mode": None,
        }
        pda_reference_df = None
        pdp_reference_df = None
        pda_current_df = None
        pdp_current_df = None
else:
    utils.log(
        "Either performance_drift key is not there in config OR is_activated is 0",
        message_run,
    )
    performance_drift_conf = {
        "monitoring_type": "performance_drift",
        "monitoring_metrics": [],
        "monitoring_mode": None,
    }
    pda_reference_df = None
    pdp_reference_df = None
    pda_current_df = None
    pdp_current_df = None

# except Exception as e:
# throw_exception(e)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Monitoring Metric JSON Template

# COMMAND ----------

monitoring_metrics_template_json = {
    "model_artifact_id": model_artifact_id,
    "input_table_id": input_table_id,
    "data_prep_deployment_job_id": data_prep_deployment_job_id,
    "deployment_master_id": deployment_master_id,
    "processing_time": int(time.time() * 1000000),
    "batch_start_id": "1",
    "batch_end_id": "10000",
    "batch_size": batch_size,
    "monitoring_type": None,
    "monitoring_sub_type": None,
    "monitoring_algo": None,
    "env": env,
    "data": None,
    "created_by_name": created_by_name,
    "created_by_id": created_by_id,
    "model_name": model_name,
    "model_version": model_version,
    "model_session_id": model_session_id,
    "project_id": project_id,
    "project_name": project_name,
    "version": version,
    "inference_metadata": str(current_table.limit(1).toPandas().to_dict("records")),
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### FEATURE DRIFT

# COMMAND ----------

# MAGIC %md
# MAGIC #### Feature Drift Inputs
# MAGIC

# COMMAND ----------

fd_reference_df = reference_df
fd_current_df = current_df

# COMMAND ----------

current_df.info()

# COMMAND ----------

print(fd_reference_df)

# COMMAND ----------

print(fd_current_df)

# COMMAND ----------

print(feature_drift_conf)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Check if Artifacts have been setup for Feature Drift Config

# COMMAND ----------

feature_monitor_algo_to_setup, listed_feature_monitors = check_for_monitor_setup(
    drift_conf=feature_drift_conf, target_type=None
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Setup Feature Drift Monitoring Artifacts if needed

# COMMAND ----------

# FIXME: Temp detection of data_type
data_type = "tabular"
try:
    for algo_dict in feature_drift_conf["monitoring_algo_overall"]:
        for key in algo_dict.keys():
            if algo_dict[key] in ["DD3", "DD4", "DD5", "FD3", "FD4", "FD5", "FD14"]:
                data_type = "timeseries"
    for algo_dict in target_drift_conf["monitoring_algo"]:
        for key in algo_dict.keys():
            if algo_dict[key] in ["DD3", "DD4", "DD5", "TD12", "TD13", "TD14", "TD15"]:
                data_type = "timeseries"

except Exception as e:
    utils.log(str(e), message_run)

# Setup Monitoring Artifacts
# setup_monitor(
#     monitor_algo_to_setup=feature_monitor_algo_to_setup,
#     drift_conf=feature_drift_conf,
#     data_type=data_type,
#     reference_df=fd_reference_df,
#     monitoring_artifact_path=monitoring_artifact_path,
# )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Remove non num features from feature drift monitoring

# COMMAND ----------

# FIXME: Random picks 1 date column for TS, need a better way to get datecol
date_column = None
feature_columns = list_numerical_columns(reference_table.select(*feature_columns))
feature_drift_conf["monitored_features"] = feature_columns


if data_type == "timeseries":
    reference_features = ast.literal_eval(feature_drift_conf["reference_features"])
    date_column_list = list_datelike_columns(
        reference_table.select(*reference_features)
    )
    if date_column_list:
        date_column = date_column_list[0]
    else:
        utils.log(
            "TIMESERIES data detected, but no date column found in reference_features!",
            message_run,
        )
utils.log(f"monitored_features {feature_columns}", message_run)
utils.log(f"date_column {date_column}", message_run)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Monitor Feature Drift for Current DF

# COMMAND ----------

# feature_drift_metrics_list = monitor_current_data(
#     drift_conf=feature_drift_conf,
#     data_type=data_type,
#     current_df=fd_current_df,
#     monitoring_algos=listed_feature_monitors,
#     monitoring_artifact_path=monitoring_artifact_path,
# )

# COMMAND ----------

try:
    feature_algo_thresholds = (
        feature_drift_conf["monitoring_algo_feature_thresholds"]
        + feature_drift_conf["monitoring_algo_overall_thresholds"]
    )

    monitored_features = feature_drift_conf["monitored_features"]

    # Sample size selection
    DATA_DRIFT_SAMPLE = 5000

    if fd_reference_df.shape[0] < DATA_DRIFT_SAMPLE:
        REF_DATA_DRIFT_SAMPLE = fd_reference_df.shape[0]
    else:
        REF_DATA_DRIFT_SAMPLE = DATA_DRIFT_SAMPLE
    if fd_current_df.shape[0] < DATA_DRIFT_SAMPLE:
        CURR_DATA_DRIFT_SAMPLE = fd_current_df.shape[0]
    else:
        CURR_DATA_DRIFT_SAMPLE = DATA_DRIFT_SAMPLE

    feature_drift_metrics_list = monitor_feature_drift_in_memory(
        data_type=data_type,
        reference_df=fd_reference_df.sample(n=REF_DATA_DRIFT_SAMPLE),
        current_df=fd_current_df.sample(n=CURR_DATA_DRIFT_SAMPLE),
        monitored_features=monitored_features,
        monitoring_algos=listed_feature_monitors,
        monitoring_algo_thresholds=feature_algo_thresholds,
        date_column=date_column,
    )
except Exception as e:
    utils.log(str(e), message_run)
    traceback.print_exc()
    feature_drift_metrics_list = []

# COMMAND ----------

print(feature_drift_metrics_list)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Apply FDR correction for DD algos

# COMMAND ----------

from monitoring.algorithms.batch_algorithms import fdr_correction

# TODO: Take Dynamic Threshold
valid_overall_algos = map_algo_id_2_name(
    [
        algo_dict["algo_id"]
        for algo_dict in feature_drift_conf["monitoring_algo_overall"]
    ]
)
feature_drift_overall_metrics_list = []
to_be_removed_index = []
for i, drift_json in enumerate(feature_drift_metrics_list):
    if drift_json["monitor_algo"] in valid_overall_algos:
        fdr_drift_json = fdr_correction(
            monitor_algo=drift_json["monitor_algo"],
            p_values=drift_json["p_val"],
            threshold=0.05,
        )
        utils.log(fdr_drift_json, message_run)
        fdr_drift_json["p_val"] = max(fdr_drift_json["p_val"])
        feature_drift_overall_metrics_list.append(fdr_drift_json)
        to_be_removed_index.append(i)
utils.log(to_be_removed_index, message_run)
feature_drift_metrics_list = [
    e for i, e in enumerate(feature_drift_metrics_list) if i not in to_be_removed_index
]

# COMMAND ----------

feature_drift_metrics_list = (
    feature_drift_metrics_list + feature_drift_overall_metrics_list
)
utils.log(
    f"feature_drift_metrics_list =====> : {feature_drift_metrics_list}", message_run
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Calculate Feature Drift Metrics JSON List

# COMMAND ----------

feature_drift_metric_json_list = []
i = 0
j = 0
for metric in feature_drift_metrics_list:
    feature_drift_metric_json = monitoring_metrics_template_json.copy()
    feature_drift_metric_json["monitoring_type"] = "feature_drift"
    if "list" in str(type(metric["is_drift"])):
        # Feature Level Drift Metric
        feature_drift_metric_json["monitoring_sub_type"] = "feature_level"
        feature_drift_metric_json["monitoring_algo"] = feature_drift_conf[
            "monitoring_algo_features"
        ][i]
        metric.pop("monitor_algo", None)
        feature_drift_metric_json["data"] = round_metrics(metric)
        print(f"FEATURE_DRIFT ======================== {feature_drift_metric_json}")
        utils.log("FEATURE_DRIFT Calculation completed", message_run)
        i = i + 1
    else:
        # Overall Drift Metric
        feature_drift_metric_json["monitoring_sub_type"] = "overall"
        feature_drift_metric_json["monitoring_algo"] = feature_drift_conf[
            "monitoring_algo_overall"
        ][j]
        metric.pop("monitor_algo", None)
        feature_drift_metric_json["data"] = round_metrics(metric)
        print(f"OVERALL_DRIFT ======================== {feature_drift_metric_json}")
        utils.log("OVERALL_DRIFT Calculation completed", message_run)
        j = j + 1
    feature_drift_metric_json_list.append(feature_drift_metric_json)

# COMMAND ----------

# MAGIC %md
# MAGIC ### TARGET DRIFT

# COMMAND ----------

# MAGIC %md
# MAGIC #### Target Drift Inputs
# MAGIC

# COMMAND ----------

print(tda_reference_df)

# COMMAND ----------

print(tda_current_df)

# COMMAND ----------

print(tdp_reference_df)

# COMMAND ----------

print(tdp_current_df)

# COMMAND ----------

print(target_drift_conf)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Check if Artifacts have been setup for Actual Target Drift Config

# COMMAND ----------

actual_target_monitor_algo_to_setup, listed_target_monitors = check_for_monitor_setup(
    target_drift_conf, target_type="actual"
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Setup Actual Target Drift Monitoring Artifacts if needed

# COMMAND ----------

data_type = "tabular"
try:
    for algo_dict in target_drift_conf["monitoring_algo"]:
        for key in algo_dict.keys():
            if algo_dict[key] in ["DD3", "DD4", "DD5", "TD12", "TD13", "TD14", "TD15"]:
                data_type = "timeseries"
except Exception as e:
    utils.log(str(e), message_run)

# Setup Monitoring Artifacts
setup_monitor(
    monitor_algo_to_setup=actual_target_monitor_algo_to_setup,
    drift_conf=target_drift_conf,
    data_type=data_type,
    reference_df=tda_reference_df,
    monitoring_artifact_path=monitoring_artifact_path,
    target=actual_target,
    target_type="actual",
    date_column=date_column,
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Monitor Actual Target Drift for Current DF

# COMMAND ----------

actual_target_drift_metrics_list = monitor_current_data(
    drift_conf=target_drift_conf,
    data_type=data_type,
    current_df=tda_current_df,
    target=actual_target,
    target_type="actual",
    monitoring_algos=listed_target_monitors,
    monitoring_artifact_path=monitoring_artifact_path,
    date_column=date_column,
)

# COMMAND ----------

print(actual_target_drift_metrics_list)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Check if Artifacts have been setup for Predicted Target Drift Config

# COMMAND ----------

(
    predicted_target_monitor_algo_to_setup,
    listed_target_monitors,
) = check_for_monitor_setup(target_drift_conf, target_type="predicted")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Setup Predicted Target Drift Monitoring Artifacts if needed

# COMMAND ----------

# Setup Monitoring Artifacts
setup_monitor(
    monitor_algo_to_setup=actual_target_monitor_algo_to_setup,
    drift_conf=target_drift_conf,
    data_type=data_type,
    reference_df=tdp_reference_df,
    monitoring_artifact_path=monitoring_artifact_path,
    target=predicted_target,
    target_type="predicted",
    date_column=date_column,
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Monitor Predicted Target Drift for Current DF

# COMMAND ----------

predicted_target_drift_metrics_list = monitor_current_data(
    drift_conf=target_drift_conf,
    data_type=data_type,
    current_df=tdp_current_df,
    target=predicted_target,
    target_type="predicted",
    monitoring_algos=listed_target_monitors,
    monitoring_artifact_path=monitoring_artifact_path,
    date_column=date_column,
)

# COMMAND ----------

print(predicted_target_drift_metrics_list)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Calculate Target Drift Metrics JSON List

# COMMAND ----------

actual_target_drift_metric_json_list = []

for i, metric in enumerate(actual_target_drift_metrics_list):
    target_drift_metric_json = monitoring_metrics_template_json.copy()
    target_drift_metric_json["monitoring_type"] = "target_drift"
    # Actual Target Drift Metric
    target_drift_metric_json["monitoring_sub_type"] = "actual"
    target_drift_metric_json["monitoring_algo"] = target_drift_conf["monitoring_algo"][
        i
    ]
    metric.pop("monitor_algo", None)
    target_drift_metric_json["data"] = round_metrics(metric)
    print(f"ACTUAL_DRIFT ========================> {target_drift_metric_json}")
    utils.log("ACTUAL_DRIFT Calculation completed", message_run)
    actual_target_drift_metric_json_list.append(target_drift_metric_json)

predicted_target_drift_metric_json_list = []
for i, metric in enumerate(predicted_target_drift_metrics_list):
    target_drift_metric_json = monitoring_metrics_template_json.copy()
    target_drift_metric_json["monitoring_type"] = "target_drift"
    # Predicted Target Drift Metric
    target_drift_metric_json["monitoring_sub_type"] = "predicted"
    target_drift_metric_json["monitoring_algo"] = target_drift_conf["monitoring_algo"][
        i
    ]
    metric.pop("monitor_algo", None)
    target_drift_metric_json["data"] = round_metrics(metric)
    print(f"PREDICTED_DRIFT ========================> {target_drift_metric_json}")
    utils.log("PREDICTED_DRIFT Calculation completed", message_run)
    predicted_target_drift_metric_json_list.append(target_drift_metric_json)

target_drift_metric_json_list = (
    actual_target_drift_metric_json_list + predicted_target_drift_metric_json_list
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### CONCEPT DRIFT

# COMMAND ----------

# MAGIC %md
# MAGIC #### Concept Drift Inputs
# MAGIC

# COMMAND ----------

print(cda_reference_df)

# COMMAND ----------

print(cdp_reference_df)

# COMMAND ----------

print(cda_current_df)

# COMMAND ----------

print(cdp_current_df)

# COMMAND ----------

print(concept_drift_conf)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Check if Artifacts have been setup for Concept Drift Config

# COMMAND ----------

concept_monitor_algo_to_setup, listed_concept_monitors = check_for_monitor_setup(
    drift_conf=concept_drift_conf, target_type=None
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Setup Concept Drift Monitoring Artifacts if needed

# COMMAND ----------

setup_monitor(
    monitor_algo_to_setup=concept_monitor_algo_to_setup,
    drift_conf=concept_drift_conf,
    data_type="tabular",
    monitoring_artifact_path=monitoring_artifact_path,
    base_y_actuals=cda_reference_df,
    base_y_predicted=cdp_reference_df,
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Monitor Concept Drift for Current DF

# COMMAND ----------

concept_drift_metrics_list = monitor_current_data(
    drift_conf=concept_drift_conf,
    data_type="tabular",
    monitoring_artifact_path=monitoring_artifact_path,
    monitoring_algos=listed_concept_monitors,
    current_y_actuals=cda_current_df,
    current_y_predicted=cdp_current_df,
)

# COMMAND ----------

print(concept_drift_metrics_list)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Calculate Concept Drift Metrics JSON List

# COMMAND ----------

concept_drift_metric_json_list = []
for i, metric in enumerate(concept_drift_metrics_list):
    concept_drift_metric_json = monitoring_metrics_template_json.copy()
    concept_drift_metric_json["monitoring_type"] = "concept_drift"
    concept_drift_metric_json["monitoring_sub_type"] = concept_drift_conf[
        "modelling_task_type"
    ]
    concept_drift_metric_json["monitoring_algo"] = concept_drift_conf[
        "monitoring_algo"
    ][i]
    metric.pop("monitor_algo", None)
    concept_drift_metric_json["data"] = round_metrics(metric)
    print(f"CONCEPT_DRIFT ========================> {concept_drift_metric_json}")
    utils.log("CONCEPT_DRIFT Calculation completed", message_run)
    concept_drift_metric_json_list.append(concept_drift_metric_json)

# COMMAND ----------

# MAGIC %md
# MAGIC ### PERFORMANCE DRIFT

# COMMAND ----------

# MAGIC %md
# MAGIC #### Performance Drift Inputs
# MAGIC

# COMMAND ----------

print(pda_reference_df)

# COMMAND ----------

print(pdp_reference_df)

# COMMAND ----------

print(pda_current_df)

# COMMAND ----------

print(pdp_reference_df)

# COMMAND ----------

print(performance_drift_conf)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Check if Artifacts have been setup for Performance Drift Config

# COMMAND ----------

performance_monitor_algo_to_setup, listed_performance_monitors = (
    check_for_monitor_setup(drift_conf=performance_drift_conf, target_type=None)
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Setup Performance Drift Monitoring Artifacts if needed

# COMMAND ----------

setup_monitor(
    monitor_algo_to_setup=performance_monitor_algo_to_setup,
    drift_conf=performance_drift_conf,
    data_type="tabular",
    monitoring_artifact_path=monitoring_artifact_path,
    base_y_actuals=pda_reference_df,
    base_y_predicted=pdp_reference_df,
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Monitor Performance Drift for Current DF

# COMMAND ----------

performance_drift_metrics_list = monitor_current_data(
    drift_conf=performance_drift_conf,
    data_type="tabular",
    monitoring_artifact_path=monitoring_artifact_path,
    monitoring_algos=listed_performance_monitors,
    current_y_actuals=pda_current_df,
    current_y_predicted=pdp_current_df,
)

# COMMAND ----------

print(performance_drift_metrics_list)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Calculate Performance Drift Metrics JSON List

# COMMAND ----------

performance_drift_metric_json_list = []
for i, metric in enumerate(performance_drift_metrics_list):
    performance_drift_metric_json = monitoring_metrics_template_json.copy()
    performance_drift_metric_json["monitoring_type"] = "performance_drift"
    performance_drift_metric_json["monitoring_sub_type"] = performance_drift_conf[
        "modelling_task_type"
    ]
    performance_drift_metric_json["monitoring_algo"] = {
        "modelling_task_name": performance_drift_conf["modelling_task_type"],
        "algo_name": "performance_metrics",
    }
    performance_drift_metric_json["data"] = round_metrics(metric)
    print(
        f"PERFORMANCE_DRIFT ========================> {performance_drift_metric_json}"
    )
    utils.log("PERFORMANCE_DRIFT Calculation completed", message_run)
    performance_drift_metric_json_list.append(performance_drift_metric_json)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Defining metrics JSON

# COMMAND ----------

drifted_monitor_types = []
aggregated_is_drift = False

for metric_json in (
    feature_drift_metric_json_list
    + target_drift_metric_json_list
    + concept_drift_metric_json_list
    + performance_drift_metric_json_list
):
    # Checking if there is a drift and adding the subtype in aggregated drift types if the monitor type has drifted
    if isinstance(metric_json["data"]["is_drift"], list) and (
        1 in metric_json["data"]["is_drift"]
    ):
        aggregated_is_drift = True
        drifted_monitor_types.append(metric_json["monitoring_type"])
    else:
        if metric_json["data"]["is_drift"] == 1:
            aggregated_is_drift = True
            drifted_monitor_types.append(metric_json["monitoring_type"])

    # Encoding JSON
    metric_json = json_encode_nans(metric_json)

    # Determination of start and end markers
    if metric_json.get("monitoring_type") == "feature_drift":
        metric_df = current_table
        start_marker = metric_df.select(F.min("id")).collect()[0][0]
        end_marker = metric_df.select(F.max("id")).collect()[0][0]
        table_name = "dpd_table"

    elif metric_json.get("monitoring_type") == "target_drift":
        if metric_json.get("monitoring_sub_type") == "actual":
            metric_df = current_table
            # metric_df = spark.createDataFrame(tda_current_df)
            start_marker = metric_df.select(F.min("id")).collect()[0][0]
            end_marker = metric_df.select(F.max("id")).collect()[0][0]
            table_name = "y_actual_Table"
        else:
            metric_df = current_table
            # metric_df = spark.createDataFrame(tdp_current_df)
            start_marker = metric_df.select(F.min("id")).collect()[0][0]
            end_marker = metric_df.select(F.max("id")).collect()[0][0]
            table_name = "inference_table"

    elif metric_json.get("monitoring_type") in ["concept_drift", "performance_drift"]:
        if metric_json.get("monitoring_type") == "concept_drift":
            # metric_df = spark.createDataFrame(curr_df_w_yact_ypred_conc)
            metric_df = current_table
        else:
            # metric_df = spark.createDataFrame(curr_df_w_yact_ypred_perf)
            metric_df = current_table
        start_marker = metric_df.select(F.min("id")).collect()[0][0]
        end_marker = metric_df.select(F.max("id")).collect()[0][0]
        table_name = "y_actual_Table"

    # Creating batch markers dict
    batch_markers = {
        "table_name": table_name,
        "start_marker": start_marker,
        "end_marker": end_marker,
    }
    metric_json["batch_markers"] = batch_markers
    metric_json["job_id"] = job_id
    metric_json["run_id"] = run_id

# COMMAND ----------

print(f"feature_drift_metric_json_list =======> {feature_drift_metric_json_list}")

# COMMAND ----------

print(f"target_drift_metric_json_list =======> {target_drift_metric_json_list}")

# COMMAND ----------

print(f"concept_drift_metric_json_list ======> {concept_drift_metric_json_list}")

# COMMAND ----------

print(f"performance_drift_metric_json_list ===> {performance_drift_metric_json_list}")

# COMMAND ----------

try:
    metrics_map = {
        "feature_drift": feature_drift_metric_json_list,
        "target_drift": target_drift_metric_json_list,
        "concept_drift": concept_drift_metric_json_list,
        "performance_drift": performance_drift_metric_json_list,
    }

    for metric_kind, metric_json in metrics_map.items():
        utils.log(metric_kind, message_run)
        if len(metric_json) > 0:
            write_metrics_to_delta(metric_json, metric_kind)
except Exception as e:
    traceback.print_exc()
    throw_exception(e)

# COMMAND ----------

# if datalake_env == "delta":
#     register_delta_as_hive(
#         db_name=f"mlcore_observability_{deployment_env}",
#         table_name=aggregated_dd_table_name,
#         dbfs_path=aggregated_dd_drift_path,
#         spark=spark,
#     )

#     register_delta_as_hive(
#         db_name=f"mlcore_observability_{deployment_env}",
#         table_name=aggregated_pd_table_name,
#         dbfs_path=aggregated_performance_drift_path,
#         spark=spark,
#     )

# COMMAND ----------

# Pushing tables in Mongo
try:
    table_types_subtypes = {
        "Monitoring_Output": ["Data_Model", "Performance"],
        "Task_Log": "Monitor_Batch",
    }

    for table_type, table_sub_type in table_types_subtypes.items():
        if isinstance(table_sub_type, list):
            for ind_table_sub_type in table_sub_type:
                if ind_table_sub_type.lower() == "data_model":
                    if (
                        metrics_map.get("feature_drift", []) == []
                        and metrics_map.get("target_drift", []) == []
                        and metrics_map.get("concept_drift", []) == []
                    ):
                        utils.log(
                            f"Since feature drift, concept drift and target drift is empty. Skip pushing table to MLCore",
                            message_run,
                        )
                        continue
                if (
                    ind_table_sub_type.lower() == "performance"
                    and metrics_map.get("performance_drift", []) == []
                ):
                    utils.log(
                        f"Since performance is empty. Skip pushing table to MLCore",
                        message_run,
                    )
                    continue

                # Checking if the table has already been created by an earlier run, if not, creating one
                is_table_present = table_exists(
                    table_type, ind_table_sub_type, job_id, version, project_id
                )

                # Calling Tables Add API
                if not is_table_present:
                    # Pushing table in Mongo
                    add_table_in_mongo(table_type, ind_table_sub_type,catalog_details)
        else:
            # Checking if the table has already been created by an earlier run, if not, creating one
            is_table_present = table_exists(
                table_type, table_sub_type, job_id, version, project_id
            )

            # Calling Tables Add API
            if not is_table_present:
                # Pushing table in Mongo
                add_table_in_mongo(table_type, table_sub_type, catalog_details)

except Exception as e:
    throw_exception(e)

# COMMAND ----------

# Pushing the aggregated tables tables in Mongo
try:
    agg_table_types_subtypes = {
        "Aggregated_Monitor_Output": ["Data_Model", "Performance"],
    }

    for table_type, table_sub_type in agg_table_types_subtypes.items():
        if isinstance(table_sub_type, list):
            for ind_table_sub_type in table_sub_type:
                # Checking if the table has already been created by an earlier run, if not, creating one
                is_table_present = table_exists(
                    table_type, ind_table_sub_type, "", "", "", True
                )

                # Calling Tables Add API
                if not is_table_present:
                    # Pushing table in Mongo
                    tables_add_payload = {
                        "name": (
                            aggregated_pd_table_name
                            if ind_table_sub_type.lower() == "performance"
                            else aggregated_dd_table_name
                        ),
                        "type": table_type,
                        "sub_type": ind_table_sub_type,
                        "deltalake_path": "",
                        "created_by_id": str(created_by_id),
                        "created_by_name": created_by_name,
                        "updated_by_id": str(created_by_id),
                        "updated_by_name": created_by_name,
                        "job_id": str(job_id),
                        "project_id": "",
                        "version": "",
                        "task_id": str(task_run_id),
                        "created_run_id":str(run_id),
                        "status": "active",
                        "primary_keys": [],
                        "workspace_id": spark.conf.get("spark.databricks.workspaceUrl"),
                        "datalake_env": platform_datalake_env
                    }
                    if platform_datalake_env == "delta":
                        if ind_table_sub_type.lower() == "performance":
                            if platform_cloud_provider == "azure":
                                tables_add_payload["dbfs_path"] = aggregated_performance_drift_path
                                tables_add_payload["db_path"] = f"{platform_db_name}.{aggregated_pd_table_name}"
                            elif platform_cloud_provider == "databricks_uc":
                                tables_add_payload["db_path"] = f"{uc_catalog_name}.{platform_db_name}.{aggregated_pd_table_name}"
                                tables_add_payload["catalog_name"] = uc_catalog_name
                        else:
                            if platform_cloud_provider == "azure":
                                tables_add_payload["dbfs_path"] = aggregated_dd_drift_path
                                tables_add_payload["db_path"] = f"{platform_db_name}.{aggregated_dd_table_name}"
                            elif platform_cloud_provider == "databricks_uc":
                                tables_add_payload["db_path"] = f"{uc_catalog_name}.{platform_db_name}.{aggregated_dd_table_name}"
                                tables_add_payload["catalog_name"] = uc_catalog_name                            
                    else:
                        table_path = (
                            aggregated_performance_drift_path
                            if ind_table_sub_type.lower() == "performance"
                            else aggregated_dd_drift_path
                        )
                        tables_add_payload["db_path"] = (
                            f"{gcp_project_id}.{bq_database_name}.{table_path}"
                        )
                    tables_add(tables_add_payload)

except Exception as e:
    traceback.print_exc()
    throw_exception(e)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Declare job as successful

# COMMAND ----------

try:
    if len(feature_drift_metric_json_list) > 0:
        update_task_log_as_per_inference_table(
            monitoring_subtype="feature_drift",
            required_entry=current_df_inference_entry,
        )
    if len(actual_target_drift_metric_json_list) > 0:
        update_task_log_as_per_inference_table(
            monitoring_subtype="target_y_actual",
            required_entry=tda_current_df_inference_entry,
        )
    if len(predicted_target_drift_metric_json_list) > 0:
        update_task_log_as_per_inference_table(
            monitoring_subtype="target_y_pred",
            required_entry=tdp_current_df_inference_entry,
        )
    if len(concept_drift_metric_json_list) > 0:
        update_task_log_as_per_inference_table(
            monitoring_subtype="concept_drift",
            required_entry=curr_df_w_yact_ypred_con_inference_entry,
        )
    if len(performance_drift_metric_json_list) > 0:
        update_task_log_as_per_inference_table(
            monitoring_subtype="performance_drift",
            required_entry=curr_df_w_yact_ypred_perf_inference_entry,
        )
except Exception as e:
    traceback.print_exc()
    throw_exception(e)

# COMMAND ----------

if any(
    feature_drift_metric_json_list
    + actual_target_drift_metric_json_list
    + predicted_target_drift_metric_json_list
    + concept_drift_metric_json_list
    + performance_drift_metric_json_list
):
    declare_job_as_successful(
        skipped_due_to_gt=null_gt_skipped_types,
        drifted_monitor_types=drifted_monitor_types,
        aggregated_is_drift=aggregated_is_drift,
    )
else:
    declare_job_as_successful(
        skipped_due_to_gt=null_gt_skipped_types,
        drifted_monitor_types=drifted_monitor_types,
        aggregated_is_drift=aggregated_is_drift,
        processed_records="no",
    )
