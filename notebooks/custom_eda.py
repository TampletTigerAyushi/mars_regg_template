# Databricks notebook source
# DBTITLE 1,Installing TigerML Package
# MAGIC %pip install --upgrade pip
# MAGIC %pip install /dbfs/FileStore/jars/MLCORE_INIT/tigerml.core-0.4.5-py3-none-any.whl --force-reinstall
# MAGIC %pip install /dbfs/FileStore/jars/MLCORE_INIT/tigerml.pyspark.core-0.4.5rc2-py3-none-any.whl
# MAGIC %pip install /dbfs/FileStore/jars/MLCORE_INIT/tigerml.eda-0.4.4-py3-none-any.whl
# MAGIC %pip install /dbfs/FileStore/jars/MLCORE_INIT/tigerml.pyspark.eda-0.4.5rc2-py3-none-any.whl
# MAGIC %pip install numpy==1.22
# MAGIC %pip install google-auth
# MAGIC %pip install google-cloud-storage
# MAGIC %pip install azure-storage-blob
# MAGIC %pip install azure-identity
# MAGIC # %pip install protobuf==3.17.2
# MAGIC %pip install sparkmeasure
# MAGIC %pip install databricks-sql-connector
# MAGIC %pip uninstall --yes cffi

# COMMAND ----------

dbutils.library.restartPython() 

# COMMAND ----------

from sparkmeasure import StageMetrics, TaskMetrics

taskmetrics = TaskMetrics(spark)
stagemetrics = StageMetrics(spark)

taskmetrics.begin()
stagemetrics.begin()

# COMMAND ----------

# DBTITLE 1,Imports
import json
import time
import traceback

import requests
from pyspark.sql import functions as F
from pyspark.sql.types import *
from requests.structures import CaseInsensitiveDict
from tenacity import retry, stop_after_attempt, stop_after_delay, wait_exponential
from tigerml.eda import TSReport
from tigerml.eda import EDAReport
from utils import utils
from utils import uc_utils
from utils.secret_mapping import SECRET_MAPPING
from sklearn.preprocessing import LabelEncoder
from databricks import sql
from pyspark.sql.types import StringType
from pyspark.ml.feature import StringIndexer
from tigerml.pyspark.core.dp import list_categorical_columns,list_datelike_columns,list_boolean_columns,list_numerical_columns
from tigerml.pyspark.core.utils import (
    append_file_to_path,
    time_now_readable,
)
import gc
from tigerml.pyspark.eda import correlation_with_target,feature_importance,feature_interactions,feature_analysis,health_analysis
from tigerml.core.reports import *
from datetime import datetime

# COMMAND ----------

# DBTITLE 1,Helper functions

def get_env_vault_scope():
    """
    Returns env and vault scope
    """
    import json

    env = (
        dbutils.notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
        .notebookPath()
        .get()
    ).split("/")[2]
    try:
        if len(dbutils.fs.ls("dbfs:/FileStore/jars/MLCORE_INIT/vault_check.json")) == 1:
            # if env == "qa":
            #     with open("/dbfs/FileStore/jars/MLCORE_INIT/vault_check_qa.json", "r") as file:
            #         vault_check_data = json.loads(file.read())
            # else:
            with open("/dbfs/FileStore/jars/MLCORE_INIT/vault_check.json", "r") as file:
                vault_check_data = json.loads(file.read())
            if "@" in env:
                return "qa", vault_check_data["client_name"]
            return env, vault_check_data["client_name"]
        else:
            return env, vault_scope
    except:
        return env, vault_scope


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


def json_str_to_pythontype(param):
    param = param.replace("'", '"')
    param = json.loads(param)

    return param


def generate_run_notebook_url(job_id, run_id):
    """
    Generates the databricks job run notebook url in runtime
    """
    workspace_url = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
    workspace_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterOwnerOrgId")
    run_notebook_url = f"{workspace_url}/?o={workspace_id}#job/{job_id}/run/{run_id}"
    return run_notebook_url


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
    return secrets_object


def get_gcp_auth_credentials(dbutils):
    import google.auth

    client_id = secrets_object.get("gcp-api-client-id", "")
    client_secret = secrets_object.get("gcp-api-client-secret", "")
    quota_project_id = secrets_object.get("gcp-api-quota-project-id", "")
    refresh_token = secrets_object.get("gcp-api-refresh-token", "")
    cred_dict = {
        "client_id": client_id,
        "client_secret": client_secret,
        "quota_project_id": quota_project_id,
        "refresh_token": refresh_token,
        "type": "authorized_user",
    }

    credentials, _ = google.auth.load_credentials_from_dict(cred_dict)

    return credentials


def __upload_blob_to_azure(
    dbutils, source_path="", file=None, target_path="", container_name="",
):
    from azure.identity import ClientSecretCredential
    from azure.storage.blob import BlobServiceClient

    try:
        TENANT_ID = secrets_object.get("az-directory-tenant", "")
        CLIENT_ID = secrets_object.get(f"az-api-client-id", "")
        CLIENT_SECRET = secrets_object.get(f"az-api-client-secret", "")
        STORAGE_ACCOUNT = secrets_object.get(f"az-storage-account", "")

        # Authentication Blob client
        credentials = ClientSecretCredential(TENANT_ID, CLIENT_ID, CLIENT_SECRET)
        service_client = BlobServiceClient(
            f"https://{STORAGE_ACCOUNT}.blob.core.windows.net", credential=credentials
        )
        container_client = service_client.get_container_client(container_name)

        if source_path:
            # Upload from local source file path to target path in Azure Blob Storage
            with open(source_path, "rb") as data:
                container_client.upload_blob(name=target_path, data=data)
        elif file:
            # Upload from in-memory file to target path in Azure Blob Storage
            container_client.upload_blob(name=target_path, data=file)

    except Exception as e:
        print(f"Error came while uploading blob object to Azure: {e}")
        raise e


def __upload_blob_to_gcp(dbutils, container_name, source_path, target_path):
    from google.cloud import storage

    try:
        credentials = get_gcp_auth_credentials(dbutils)
        project = secrets_object.get( "gcp-api-quota-project-id", "")

        # Use the obtained credentials to create a client to interact with GCP services
        storage_client = storage.Client(credentials=credentials, project=project)

        bucket_client = storage_client.bucket(container_name)

        # Upload the model file to GCS
        blob = bucket_client.blob(target_path)
        blob.upload_from_filename(source_path)

    except Exception as e:
        print(f"Error came while uploading blob object from gcp : {e}")
        raise e


def upload_blob_to_cloud(**kwargs):
    """
    Upload the blob from the cloud storage.

    This function will help upload the blob from the cloud storage service like Azure, AWS, GCP.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments containing operation details, including the `resource_type`.

    Returns
    -------
    The result of the dispatched operation based on the `resource_type`.

    Notes
    -----
    - The function loads the blob object from cloud storage with below parameters :
    - For Azure
        - 'dbutils': The dbutils object to retrive the secrets needed for the APIs.
        - 'container_name': The container where the blob object is stored.
        - 'blob_path': The local file path where the blob is present.
        - 'target_path' : The target path where the blob has to be downloaded.

    - For GCP
        - 'dbutils': The dbutils object to retrive the secrets needed for the APIs.
        - 'container_name': The bucket where the blob  object is stored.
        - 'blob_path': The local file path where the blob is present.
        - 'target_path' : The target path where the blob has to be downloaded.

    - It is essential to provide the correct `resource_type`. Currently supported resources are : az, gcp
    """
    resource_type = kwargs.get("resource_type", None)
    if not resource_type or resource_type in [""]:
        raise Exception("Resource type is not passed or is empty.")

    del kwargs["resource_type"]  # Delete the key since it will not be used by modules

    if resource_type not in ["az", "gcp", "azure"]:
        raise Exception(f"Uploading blob object from {resource_type} is not supported.")

    if resource_type.lower() in ["az", "azure"]:
        return __upload_blob_to_azure(**kwargs)

    if resource_type.lower() == "gcp":
        return __upload_blob_to_gcp(**kwargs)


def get_cluster_info(spark):
    p = "spark.databricks.clusterUsageTags."
    conf = spark.sparkContext.getConf().getAll()
    conf_dict = {k.replace(p, ""): v for k, v in conf if k.startswith(p)}

    return conf_dict


def get_catalog_details(env):
    h1 = get_headers(vault_scope)
    response = requests.get(
        API_ENDPOINT + GET_CATALOG + env, headers=h1
    )
    print(
        f"\n\
    Logging task:\n\
    endpoint - {GET_CATALOG}\n\
    status   - {response}\n\
    response - {response.text}"
    )

    return response


# COMMAND ----------

# DBTITLE 1,API Authorization
# Fetching env and scope information
message_run = []
message_task = []

# Fetching env and scope information
env, vault_scope = get_env_vault_scope()
secrets_object = fetch_secrets_from_dbutils(dbutils, message_run)
h1 = get_headers(vault_scope)
try:
    API_ENDPOINT = dbutils.widgets.get("tracking_base_url")
except:
    API_ENDPOINT = get_app_url()
az_container_name = secrets_object.get("az-container-name", "")
az_storage_account = secrets_object.get("az-storage-account", "")
uc_container_name = secrets_object.get("uc-container-name", "")
uc_volume_name = secrets_object.get("uc-volume-name", "")

# COMMAND ----------

# DBTITLE 1,Global Variables
job_id, run_id, task_run_id, taskKey = utils.get_job_details(dbutils)
run_notebook_url = generate_run_notebook_url(job_id, run_id)
source_info = utils.get_params("source_info", dbutils)
ground_truth_table_id = dbutils.widgets.get("ground_truth_table_id")
ground_truth_table_primary_keys = json_str_to_pythontype(
    dbutils.widgets.get("ground_truth_table_primary_keys")
)
input_table_primary_keys = json_str_to_pythontype(
    dbutils.widgets.get("input_table_primary_keys")
)

deployment_env = dbutils.widgets.get("deployment_env")
JOB_TASK_ADD = "mlapi/job/task/log/add"
JOB_TASK_UPDATE = "mlapi/job/task/log/update"
JOB_RUNS_UPDATE = "mlapi/job/runs/log/update"
MEDIA_ARTIFACTS_ADD = "mlapi/add_media_artifacts"
GET_CATALOG = "mlapi/get_catalog?deployment_env="

# Source Information
project_id = source_info["project_id"]
project_name = source_info["project_name"]
version = source_info["version"]
feature_table_id = source_info["table_id"]
created_by_id = source_info["created_by_id"]
created_by_name = source_info["created_by_name"]
date_column = source_info["date_column"]
target_column = source_info["target_column"]
if target_column == "None":
    target_column = None

select_full_df = str(source_info.get("select_all", "no")) == "yes"

try:
    datalake_env = dbutils.widgets.get("datalake_env").lower()
except Exception as e:
    utils.log(f"Exception while retrieving data lake environment : {e}", message_run)
    datalake_env = "delta"

message_run = []
message_task = []

try:
    cloud_provider = dbutils.widgets.get("cloud_provider").lower()
except:
    cloud_provider = "gcp"
    utils.log(f"Exception while retrieving Cloud Provider : {e}", message_run)

try:
    from tigerml.core.reports.html import report_configs
    ml_engine = dbutils.widgets.get("ml_engine").lower()
    report_metadata = utils.get_params("report_metadata", dbutils)
    report_color = report_metadata["report_color"]
    report_configs(custom_text=f"Generated by {ml_engine}",custom_background=report_color)
except Exception as e:
    from tigerml.core.reports.html import report_configs
    utils.log(f"Exception while retrieving reportconfigs : {e}", message_run)
    report_configs()

# COMMAND ----------

# DBTITLE 1,API helper methods

def task_log_data(message_run, message_task):

    ts = int(time.time() * 1000000)

    task_log_data = {
        "project_id": source_info["project_id"],
        "version": source_info["version"],
        "job_id": str(job_id),
        "run_id": str(run_id),
        "task_id": str(task_run_id),
        "status": "running",
        "start_time": ts,
        "created_by_id": source_info["created_by_id"],
        "created_by_name": source_info["created_by_name"],
        "job_type": "EDA",
    }
    h1 = get_headers(vault_scope)
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

    t = str(int(time.time() * 1000000))

    return message_run, message_task


def job_runs_update(message_run, message_task, status, message):
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

    ts = str(int(time.time() * 1000000))
    message_run.append(
        {
            "time": ts,
            "message": "Job with " + str(job_id) + " and " + str(run_id) + message,
        }
    )
    message_task.append(
        {
            "time": ts,
            "message": taskKey
            + " within "
            + str(job_id)
            + " and "
            + str(run_id)
            + message,
        }
    )
    log_data = {
        "project_id": source_info["project_id"],
        "version": str(source_info["version"]),
        "job_id": str(job_id),
        "run_id": str(run_id),
        "task_id": str(task_run_id),
        "end_time": str(ts),
        "job_type": "EDA",
        "status": status,
        "message": message_run,
        "eda_logs": {},
        "updated_by_id": source_info["created_by_id"],
        "updated_by_name": source_info["created_by_name"],
        "parent_table_id": source_info.get("table_id", None),
        "target_column": source_info.get("target_column", None),
        "job_type": "EDA",
        "cpu": str(aggregate_compute_metrics.get("executorCpuTime", "NA")),
        "ram": str(aggregate_compute_metrics.get("peakExecutionMemory", "NA")),
        "cluster_info": get_cluster_info(spark),
        "compute_metrics": compute_metrics,
        "run_notebook_url": run_notebook_url,
        "media_artifacts_path": (
            report_directory.split("/custom_reports")[0]
            if "report_directory" in globals()
            else ""
        ),
    }
    # Log the payload excluding large fields
    log_payload = {
        key: value
        for key, value in log_data.items()
        if key not in ["compute_metrics", "cluster_info", "message"]
    }

    print(f"API URL : {JOB_RUNS_UPDATE}. Payload : {log_payload}")

    h1 = get_headers(vault_scope)
    response = requests.put(API_ENDPOINT + JOB_RUNS_UPDATE, json=log_data, headers=h1)

    print(
        f"\n\
    Logging run '{message}': \n\
    endpoint - {JOB_RUNS_UPDATE}\n\
    status   - {response}\n\
    response - {response.text}",
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

    return message_run, message_task


def job_task_update(message_run, message_task, status, run_notebook_url):

    ts = str(int(time.time() * 1000000))
    task_log_data = {
        "job_id": str(job_id),
        "run_id": str(run_id),
        "task_id": str(task_run_id),
        "end_time": str(ts),
        "status": status,
        "message": message_task,
        "updated_by_id": source_info["created_by_id"],
        "updated_by_name": source_info["created_by_name"],
        "job_type": "EDA",
        "run_notebook_url": run_notebook_url,
    }
    h1 = get_headers(vault_scope)
    response = requests.put(
        API_ENDPOINT + JOB_TASK_UPDATE, json=task_log_data, headers=h1
    )

    utils.log(
        f"\n\
    Logging task '{status}': \n\
    endpoint - {JOB_TASK_UPDATE}\n\
    status   - {response}\n\
    response - {response.text}\n\
    payload  - {task_log_data}\n",
        message_run,
    )

    return message_run, message_task


def take_random_rows(df, n=20000):
    """
    Samples n rows randomly from the input dataframe
    """
    fraction_value = n / df.count()
    if fraction_value > 1:
        return df
    else:
        return df.sample(withReplacement=False, fraction=fraction_value, seed=2022)

# Check if the target column is categorical
def is_categorical(df, column):
    
    try:
        # Get the data type of the column
        dtype = df.schema[column].dataType

        # Check if the type is StringType
        if isinstance(dtype, StringType):
            return True
        else:
            # Optionally, check if a numerical column has a small number of unique values
            unique_count = df.select(column).distinct().count()
            total_count = df.count()
            # Define a threshold for unique values to consider as categorical
            threshold = 0.05 * total_count  # Adjust threshold as needed
            return unique_count <= threshold
    except Exception as e:
            traceback.print_exc()
            return None
    
    

@retry(
    wait=wait_exponential(min=4, multiplier=1, max=10),
    stop=(stop_after_delay(40) | stop_after_attempt(5)),
)
def get_delta_table(table_id):
    response = requests.get(
        API_ENDPOINT + f"mlapi/tables/metadata/list?table_id={table_id}", headers=h1
    )
    utils.log(
        f"\n\
    endpoint - mlapi/tables/metadata/list?table_id={table_id}\n\
    response - {response}",
        message_run,
    )
    if response.status_code not in [200, 201]:
        raise Exception(
            f"API Error : The get_delta_table API returned {response.status_code} response."
        )
    return response
def media_artifacts_add(cloud_provider,model_artifact_id ="",entity_type=""):

    ts = int(time.time() * 1000000)

    artifacts_data = {
        "project_id": project_id,
        "version": version,
        "job_id": di_job_id,
        "run_id": di_run_id,
        "folder_type": "reports",
        "sub_folder_path": "custom_reports",
        "media_artifacts_path": target_path,
        "model_artifact_id": model_artifact_id,
        "entity_type": entity_type,
    }

    # Add additional parameters based on cloud provider
    if cloud_provider.lower() == 'azure':
        artifacts_data["container_name"] = container_name
        artifacts_data["az_storage_account"] = az_storage_account

    elif cloud_provider.lower() == 'gcp':
        artifacts_data["container_name"] = container_name
        artifacts_data["gcp_project_id"] = quota_project_id

    elif cloud_provider.lower() == 'databricks_uc':
        catalog_details = get_catalog_details(deployment_env).json().get("data")[0]
        artifacts_data["catalog_name"] = catalog_details["catalog_name"]
        artifacts_data["schema_name"] = catalog_details["catalog_schema_name"]
        artifacts_data["volume_name"] = catalog_details["volume_name"] 

    h1 = get_headers(vault_scope)
    response = requests.post(
        API_ENDPOINT + MEDIA_ARTIFACTS_ADD, json=artifacts_data, headers=h1
    )
    utils.log(
        f"\n\
    Logging task:\n\
    endpoint - {MEDIA_ARTIFACTS_ADD}\n\
    status   - {response}\n\
    response - {response.text}\n\
    payload  - {artifacts_data}\n",
        message_run,
    )

    t = str(int(time.time() * 1000000))

    return response

# COMMAND ----------


def get_table_data():

    get_table_response = get_delta_table(feature_table_id).json()["data"][0]
    di_job_id = get_table_response.get("job_id", "")
    di_run_id = get_table_response.get("created_run_id", "")
    try:
        start_date = dbutils.widgets.get("start_date")
    except:
        start_date = ""
    try:
        end_date = dbutils.widgets.get("end_date")
    except:
        end_date = ""
    try:
        num_records = dbutils.widgets.get("num_records")
    except:
        num_records = ""   

    features_df = uc_utils.read_data(
        spark=spark,
        sql=sql,
        dbutils=dbutils,
        vault_scope=vault_scope,
        api_endpoint=API_ENDPOINT,
        headers=h1,
        table_id=feature_table_id
    )

    if ground_truth_table_id and ground_truth_table_id != "":
        gt_table_df = uc_utils.read_data(
            spark=spark,
            sql=sql,
            dbutils=dbutils,
            vault_scope=vault_scope,
            api_endpoint=API_ENDPOINT,
            headers=h1,
            table_id=ground_truth_table_id
        )
    # IF target is available in feature table, drop it.
    if source_info["target_column"] in features_df.columns:
        utils.log("Dropping target from Feature Table", message_run)
        features_df = features_df.drop(source_info["target_column"])

    # IF date column is available in GroundTruth Table, drop it
    if (
        ground_truth_table_id
        and ground_truth_table_id != ""
        and date_column in gt_table_df.columns
    ):
        utils.log("Dropping date column from GT Table", message_run)
        gt_table_df = gt_table_df.drop(date_column)

    # Join GT and Feature Table
    if ground_truth_table_id and ground_truth_table_id != "":
        if input_table_primary_keys and len(input_table_primary_keys) > 0:
            features_df = features_df.join(
                gt_table_df.select(*input_table_primary_keys, target_column),
                on=input_table_primary_keys,
            )
        else:
            features_df = features_df.join(gt_table_df)

    if not select_full_df:
        if start_date and end_date : 
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
            features_df = features_df.filter(F.col("date") >= (start_date)).filter(F.col("date") <= (end_date))      
        if num_records: 
            features_df = features_df.limit(int(num_records))
    
    if not any([select_full_df, start_date, end_date, num_records]):
        features_df = take_random_rows(features_df)

    return features_df, di_job_id, di_run_id



# COMMAND ----------

# MAGIC %md
# MAGIC ## Function to perform custom changes

# COMMAND ----------

def customeda(df):
    #input spark dataframe
    df=df.toPandas()
    eda_report=EDAReport(df)
    
    return eda_report
    

# COMMAND ----------

# DBTITLE 1,EDA Report Generation
try:

    utils.log("Logging task...", message_run)
    message_run, message_task = task_log_data(message_run, message_task)

    utils.log(
        f"\
    Source info: \n\
    project_id   : {project_id} \n\
    version      : {version} \n\
    date_column  : {date_column} \n\
    ground_truth_table_id : {ground_truth_table_id}\n\
    ground_truth_table_primary_keys : {ground_truth_table_primary_keys}\n\
    feature_table_id : {feature_table_id}\n\
    input_table_primary_keys : {input_table_primary_keys}\n\
    created_by_id : {created_by_id}\n\
    created_by_name : {created_by_name}\n\
    target_column: {target_column}\n\
    datalake_env : {datalake_env}\n\
    cloud_provider : {cloud_provider}\n\
    ",
        message_run,
    )

    df, di_job_id, di_run_id = get_table_data()

    utils.log(
        f"\
    Reading source table:\n\
    rows: {df.count()}\n\
    columns: {len(df.columns)}\n",
        message_run,
    )

    report_directory = f"{env}/media_artifacts/{project_id}/{version}/{di_job_id}/{di_run_id}/custom_reports"
    utils.log(f"Report Directory : {report_directory}", message_run)

    ## Drop reserved columns
    reserved_columns = [
        "dataset_type_71E4E76EB8C12230B6F51EA2214BD5FE",
        "partition_id_71E4E76EB8C12230B6F51EA2214BD5FE",
        "timestamp",
        "date",
        "id",
    ]
    if "id" in input_table_primary_keys:
        reserved_columns.remove("id")
    df = df.drop(*reserved_columns)

    is_timeseries_eda = False

    # Configure Time Series EDA charts if date_column is passed.
    if date_column and date_column != "":
        utils.log(f"Input Date Column : {date_column}", message_run)
        is_timeseries_eda = True
        if "partition_name_71E4E76EB8C12230B6F51EA2214BD5FE" not in df.columns:
            utils.log(
                f"Since the partition_name_71E4E76EB8C12230B6F51EA2214BD5FE column is not present, adding it with value 'Overall'",
                message_run,
            )
            df = df.withColumn(
                "partition_name_71E4E76EB8C12230B6F51EA2214BD5FE", F.lit("Overall")
            )
        utils.log("Configured the time series chart", message_run)

    head_start_time = time.time()

    if target_column : 
        if dict(df.dtypes)[target_column] == 'string':
            # Use StringIndexer to convert the string column to numeric indices
            indexer = StringIndexer(inputCol=target_column, outputCol=target_column + "_index")
            df = indexer.fit(df).transform(df).drop(target_column).withColumnRenamed(target_column + "_index", target_column)

    if cloud_provider.lower() == "gcp":
        container_name = f"{az_container_name}_dev"
    else:
        container_name = az_container_name

    if is_timeseries_eda:
        df = df.toPandas()
        report_path = f"/dbfs/FileStore/TsEdaReport_{int(time.time())}.html"
        utils.log(f"Temporary report path : {report_path}", message_run)
        eda_report = TSReport(df, ts_column=date_column, y=target_column)
        eda_report.get_report(y=target_column, save_path=report_path)
        if cloud_provider.lower() == "databricks_uc":
            catalog_details = get_catalog_details(deployment_env).json()["data"][0]
            target_path=f'/Volumes/{catalog_details["catalog_name"]}/{catalog_details["catalog_schema_name"]}/{catalog_details["volume_name"]}/{report_directory}/TsEdaReport_{int(time.time())}.html'
            dbutils.fs.cp("dbfs:" + report_path.split("/dbfs")[-1], target_path)
        else:
            target_path = f"{report_directory}/TsEdaReport_{int(time.time())}.html"
            utils.log(
                f"report path in {cloud_provider} storage : {report_directory}",
                message_run,
            )
            upload_blob_to_cloud(
                container_name=container_name,
                source_path=report_path,
                dbutils=dbutils,
                target_path=target_path,
                resource_type=cloud_provider,
            )

    else:
        report_path = f"/dbfs/FileStore/EdaReport_{int(time.time())}.html"
        utils.log(f"Temporary report path : {report_path}", message_run)
        # Dropping cat_columns who have >25 unique values at it can break key_drivers in sparkEDA
        cat_columns = list_categorical_columns(df) + list_datelike_columns(df)
        for catcol in cat_columns:
            if df.select(catcol).distinct().count() > 25:
                df = df.drop(catcol)
        # if target_column : 
        #     eda_report = EDAReportPyspark(data=df, is_classification=is_categorical(df,target_column), y=target_column)
        # else :
        #     eda_report = EDAReportPyspark(data=df,y=target_column)
        eda_report=customeda(df)
        eda_report.get_report(y=target_column, save_path=report_path)
        if cloud_provider.lower() == "databricks_uc":
            catalog_details = get_catalog_details(deployment_env).json()["data"][0]
            target_path=f'/Volumes/{catalog_details["catalog_name"]}/{catalog_details["catalog_schema_name"]}/{catalog_details["volume_name"]}/{report_directory}/EdaReport_{int(time.time())}.html'
            dbutils.fs.cp("dbfs:" + report_path.split("/dbfs")[-1], target_path)
        else:
            target_path=f"{report_directory}/EdaReport_{int(time.time())}.html"
            utils.log(
                f"report path in {cloud_provider} storage : {report_directory}",
                message_run,
            )
            upload_blob_to_cloud(
                container_name=container_name,
                source_path=report_path,
                dbutils=dbutils,
                target_path=target_path,
                resource_type=cloud_provider,
            )
    Response =  media_artifacts_add(cloud_provider)
    dbutils.fs.rm("dbfs:" + report_path.split("/dbfs")[-1], True)

    utils.log(f"EDA run time: {round(time.time() - head_start_time, 2)}", message_run)

    ## Logging Success
    message_run, message_task = job_runs_update(
        message_run, message_task, "success", " success"
    )
    message_run, message_task = job_task_update(
        message_run, message_task, "success", run_notebook_url
    )

except Exception as e:

    ## Logging Failure
    message_run, message_task = job_runs_update(
        message_run, message_task, "failed", f" failed with exception {str(e)}."
    )
    message_run, message_task = job_task_update(
        message_run, message_task, "failed", run_notebook_url
    )
    traceback.print_exc()

# COMMAND ----------


