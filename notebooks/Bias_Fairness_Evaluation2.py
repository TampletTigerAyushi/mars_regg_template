# Databricks notebook source
# MAGIC %pip install --upgrade pip
# MAGIC %pip install /dbfs/FileStore/jars/MLCORE_INIT/tigerml.core-0.4.4-py3-none-any.whl --force-reinstall
# MAGIC %pip install aequitas==0.42.0
# MAGIC %pip install google-auth
# MAGIC %pip install google-cloud-storage
# MAGIC %pip install azure-storage-blob
# MAGIC %pip install azure-identity
# MAGIC # %pip install protobuf==3.17.2
# MAGIC %pip install databricks-sql-connector

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from aequitas.preprocessing import preprocess_input_df
from tigerml.core.reports import create_report
from utils import utils
from utils import uc_utils
from databricks import sql
import numpy as np
import pandas as pd

import json
from utils.vault_scope import VAULT_SCOPE
from utils.secret_mapping import SECRET_MAPPING, CONFIGS
import requests
from requests.structures import CaseInsensitiveDict

# COMMAND ----------

def get_env_vault_scope():
    """
    Returns env and vault scope
    """
    try:
        env = dbutils.widgets.get("env")
    except:
        env = (
            dbutils.notebook.entry_point.getDbutils()
            .notebook()
            .getContext()
            .notebookPath()
            .get()
        ).split("/")[2]

    return env, VAULT_SCOPE.get(env, {}).get("client_name", "")


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


def get_gcp_auth_credentials(dbutils):
    import google.auth

    _, vault_scope = get_env_vault_scope()
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
        - 'container_namecontainer_name': The bucket where the blob  object is stored.
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


def media_artifacts_add(cloud_provider,model_artifact_id="",entity_type=""):

    ts = int(time.time() * 1000000)

    artifacts_data = {
        "project_id": project_id,
        "version": version,
        "job_id": job_id,
        "run_id": run_id,
        "folder_type": "reports",
        "sub_folder_path": "Bias_Evaluation",
        "media_artifacts_path": target_path,
        "model_artifact_id": model_artifact_id,
        "entity_type": entity_type
    }
    # Add additional parameters based on cloud provider
    if cloud_provider.lower() == 'azure':
        artifacts_data["container_name"] = container_name
        artifacts_data["az_storage_account"] = az_storage_account
    elif cloud_provider.lower() == 'gcp':
        artifacts_data["container_name"] = container_name
        artifacts_data["gcp_project_id"] = quota_project_id
    elif cloud_provider.lower() == 'databricks_uc':
        catalog_details = get_catalog_details(env).json().get("data")[0]
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

import json


def json_str_to_list(param):
    param = param.replace("'", '"')
    param = json.loads(param)

    return param

message_run = []
message_task = []

env, vault_scope = get_env_vault_scope()
secrets_object = fetch_secrets_from_dbutils(dbutils, message_run)

try:
    API_ENDPOINT = dbutils.widgets.get("tracking_base_url")
except:
    API_ENDPOINT = get_app_url()

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

h1 = get_headers(vault_scope)

MEDIA_ARTIFACTS_ADD = "mlapi/add_media_artifacts"
GET_CATALOG = "mlapi/get_catalog?deployment_env="

# COMMAND ----------

report_directory = dbutils.widgets.get("report_directory")
reserved_columns = ["prediction"]
train_table_id = dbutils.widgets.get("train_table_id")
features = dbutils.widgets.get("feature_columns").split(",")
target = [dbutils.widgets.get("target_columns")]
datalake_env = dbutils.widgets.get("datalake_env").lower()
cloud_provider = dbutils.widgets.get("cloud_provider")
sensitive_columns = dbutils.widgets.get("sensitive_variables").split(",")
modelling_task_type = dbutils.widgets.get("modelling_task_type")
model_artifact_id = dbutils.widgets.get("model_artifact_id")
project_id = dbutils.widgets.get("project_id")
version = dbutils.widgets.get("version")
job_id = dbutils.widgets.get("job_id")
run_id = dbutils.widgets.get("run_id")
env = dbutils.widgets.get("env")

# COMMAND ----------

bias_check_df = uc_utils.read_data(
    spark=spark,
    sql=sql,
    dbutils=dbutils,
    vault_scope=vault_scope,
    api_endpoint=API_ENDPOINT,
    headers=h1,
    table_id=train_table_id
)

# COMMAND ----------

if not sensitive_columns:
    sensitive_columns = detect_categorical_cols(bias_check_df.select(features))
total_records = bias_check_df.count()

# COMMAND ----------

if modelling_task_type.lower() == "classification":
    df = bias_check_df.toPandas()[sensitive_columns + target + ["prediction"]]
else:
    df = bias_check_df.toPandas()
    threshold_target = df[target].median()
    threshold_prediction = df["prediction"].median()
    df[target] = (df[target] >= threshold_target).astype(int)
    df["prediction"] = (df["prediction"] >= threshold_prediction).astype(int)
    df = df[sensitive_columns + target + ["prediction"]]

# COMMAND ----------

import pandas as pd
import numpy as np

def compute_fairness_metrics(df, sensitive_vars, prediction_col, target_col):
    metrics_by_sensitive_var = []

    for sensitive_var in sensitive_vars:
        unique_groups = df[sensitive_var].unique()

        for group in unique_groups:
            # Filter dataset for the current group
            df_group = df[df[sensitive_var] == group]

            # Calculate mean predictions for the group and overall
            mean_pred_group = np.mean(df_group[prediction_col])
            mean_pred_all = np.mean(df[prediction_col])

            # Calculate mean target for the group and overall
            mean_target_group = np.mean(df_group[target_col])
            mean_target_all = np.mean(df[target_col])

            # Compute fairness metrics
            mean_difference = mean_pred_group - mean_pred_all
            disparate_impact = mean_pred_group / mean_pred_all

            metrics_by_sensitive_var.append({
                'Sensitive Variable': sensitive_var,
                'Group': group,
                'Mean Difference': mean_difference,
                'Disparate Impact': disparate_impact,
                'Mean Target Group': mean_target_group,
                'Mean Target All': mean_target_all
                # Add more metrics as needed
            })

    # Convert metrics to DataFrame for reporting
    metrics_df = pd.DataFrame(metrics_by_sensitive_var)

    return metrics_df

# Define sensitive variables, prediction column, and target column
sensitive_vars =sensitive_columns
prediction_col = 'prediction'
target_col = target

# Compute bias and fairness metrics
fairness_report = compute_fairness_metrics(df, sensitive_vars, prediction_col, target_col)
fairness_report=pd.DataFrame(fairness_report)
# Display fairness report
print("Fairness Report:")
print(fairness_report)


# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------



# Plotting Mean Difference by Sensitive Variable and Group
plt.figure(figsize=(10, 6))
sns.barplot(x='Group', y='Mean Difference', hue='Sensitive Variable', data=fairness_report)
plt.title('Mean Difference by Sensitive Variable and Group')
plt.xlabel('Group')
plt.ylabel('Mean Difference')
plt.xticks(rotation=45)
plt.legend(title='Sensitive Variable')
fg1=plt.tight_layout()
plt.show()
plt.savefig('mean_difference_plot.png')

# Plotting Disparate Impact by Sensitive Variable and Group
plt.figure(figsize=(10, 6))
sns.barplot(x='Group', y='Disparate Impact', hue='Sensitive Variable', data=fairness_report)
plt.title('Disparate Impact by Sensitive Variable and Group')
plt.xlabel('Group')
plt.ylabel('Disparate Impact')
plt.xticks(rotation=45)
plt.legend(title='Sensitive Variable')
fg2=plt.tight_layout()
plt.show()
plt.savefig('disparate_impact_plot.png')

# Close all figures to prevent displaying them inline again
plt.close('all')



# COMMAND ----------


# Constructing the report dictionary
report = {
    "Category Level Bias": {
        "Mean Difference Plot": (fg1, "<p style='text-align:left;'><strong>Summary:</strong> The bar plot above shows the Mean Difference across different groups defined by sensitive variables (fuel).</p>"),
    },
    "Group Level Bias": {
        "Disparate Impact Plot": (fg2, "<p style='text-align:left;'><strong>Summary:</strong> The bar plot above displays the Disparate Impact across different groups defined by sensitive variables (fuel).</p>")
    }
}

# COMMAND ----------

for section, plots in report.items():
    print(f"{section}:")
    for plot_name, (plot_file, plot_desc) in plots.items():
        print(f"  {plot_name}:")
        print(f"    Plot File: {plot_file}")
        print(f"    Description: {plot_desc}")
    print()

# COMMAND ----------

import time

# COMMAND ----------

report_name = f"BiasReport_{int(time.time())}"
report_path = f"/dbfs/FileStore/MONITORING/bias_report/{report_name}"
create_report(
    report,
    name=report_path,
    format=".html",
    columns=1,
)

# COMMAND ----------




if cloud_provider.lower() == "databricks_uc":
    catalog_details = get_catalog_details(env).json()["data"][0]
    target_path=f'/Volumes/{catalog_details["catalog_name"]}/{catalog_details["catalog_schema_name"]}/{catalog_details["volume_name"]}/{report_directory}/Bias_Evaluation/BiasReport_{int(time.time())}.html'
    dbutils.fs.cp(f"dbfs:{report_path.split('/dbfs')[-1]}.html",target_path)
else:
    target_path = f"{report_directory}/Bias_Evaluation/BiasReport_{int(time.time())}.html"
    upload_blob_to_cloud(
        container_name=container_name,
        source_path=f"{report_path}.html",
        dbutils=dbutils,
        target_path=target_path,
        resource_type=cloud_provider,
    )
Response =  media_artifacts_add(cloud_provider,model_artifact_id = model_artifact_id,entity_type="bias")
dbutils.fs.rm(f"dbfs:{report_path.split('/dbfs')[-1]}.html", True)
print(f"report_directory : {report_directory}")
