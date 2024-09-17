import os
import sys
import time
import yaml
import requests
import json
from requests.structures import CaseInsensitiveDict

API_BASE_URL = sys.argv[1]
AZ_CLIENT_ID = sys.argv[2]
AZ_CLIENT_SECRET = sys.argv[3]
AZ_TENANT = sys.argv[4]
ENV = sys.argv[5]
GIT_REPO_URL = sys.argv[6]
COMMIT_ID = sys.argv[7]
TRIGGERED_BY = sys.argv[8]
PIPELINE_NAME = sys.argv[9]
PIPELINE_ID = sys.argv[10]
PIPELINE_RUN_ID = sys.argv[11]
TARGET_BRANCH = sys.argv[12]
DEVOPS_PROJECT_NAME = sys.argv[13]
DEVOPS_ORG_URL = sys.argv[14]
DATABRICKS_REPO_FOLDER_NAME = sys.argv[15]
DATABRICKS_HOST = sys.argv[16]
DEVOPS_TOKEN = sys.argv[17]
TRAIN_META_JOB_TYPE =  sys.argv[18]
RETRAIN_META_JOB_TYPE = sys.argv[19]
model_name =  sys.argv[20]
inference_job_name =  model_name+"_"+"inference"
api_trigger = sys.argv[21]
model_approval_id =  sys.argv[22]
model_artifact_id  = sys.argv[23]


if ENV == "qa":
    DATABRICKS_REPO_FOLDER_NAME = f"{DATABRICKS_REPO_FOLDER_NAME}_QA"
    
DEVOPS_ORG_NAME = DEVOPS_ORG_URL.split("/")[-2]
TARGET_BRANCH = TARGET_BRANCH.replace("refs/heads/", "")

file_path = "azure-db-repos-pipelinesv3/utility/job_ids.json"

try:
    with open(file_path, "r") as json_file:
        job_ids = json.load(json_file)
    inference_job_create_status =  True
except Exception as e:
    inference_job_create_status =  False

print(job_ids)
def get_access_tokens(client_id, scope, client_secret, tenant_id):
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
    url = (
        "https://login.microsoftonline.com/" + tenant_id + "/oauth2/v2.0/token"
    )
    resp = requests.post(url, headers=headers, data=data).json()
    token = resp["access_token"]
    token_string = "Bearer" + " " + token
    return token_string

def get_headers(client_id, client_secret, tenant_id):
    h1 = CaseInsensitiveDict()
    scope = client_id + "/.default"
    h1["Authorization"] = get_access_tokens(
        client_id, scope, client_secret, tenant_id
    )
    h1["Content-Type"] = "application/json"
    return h1

def get_project_details(API_BASE_URL,ENV,Header):
    if os.path.exists(f"data_config/SolutionConfig_{ENV}.yaml"):
        yaml_file_path = f"data_config/SolutionConfig_{ENV}.yaml"
    else:
        yaml_file_path = "data_config/SolutionConfig.yaml"
    with open(yaml_file_path, 'r') as file:
        yaml_content = file.read()
    config = yaml.safe_load(yaml_content)
    session_id = config.get("general_configs").get("sdk_session_id").get(f'{ENV}')
    print(Header)
    url = f"https://{API_BASE_URL}/mlapi/get_transformation_session?dag_session_id={session_id}"
    try:
        print(f"url : {url}")
        response = requests.get(
            url,
            headers=Header,
        )
        return response.json()
    except Exception as e:
        print(e)
        return None
    
def get_devops_folder_name(DEVOPS_TOKEN):
    url = f"https://dev.azure.com/{DEVOPS_ORG_NAME}/{DEVOPS_PROJECT_NAME}/_apis/pipelines/{PIPELINE_ID}?api-version=7.1-preview.1"
    try:
        response = requests.get(url, auth=("", DEVOPS_TOKEN))
        result = response.json()["folder"][1:]
        return result
    except Exception as e:
        print(e)
    


def devops_data_add(API_BASE_URL,PAYLOAD,Header):
    url = f"https://{API_BASE_URL}/mlapi/devops/data/add"
    try:
        print(f"url : {url}")
        print(f"payload : {PAYLOAD}")
        status = requests.post(
            url,
            json=PAYLOAD,
            headers=Header,
        )
        print(f"status code {status}")
        print(status.json())
    except Exception as e:
        print(e)

def register_ext_metajob(API_BASE_URL,PAYLOAD,Header):
    url = f"https://{API_BASE_URL}/mlapi/register_ext_job"
    try:
        print(f"url : {url}")
        print(f"payload : {PAYLOAD}")
        status = requests.post(
            url,
            json=PAYLOAD,
            headers=Header,
        )
        print(f"status code {status}")
        print(status.json())
    except Exception as e:
        print(e)

#print(job_ids)

Header = get_headers(AZ_CLIENT_ID, AZ_CLIENT_SECRET, AZ_TENANT)

session_info = get_project_details(API_BASE_URL,ENV,Header)
session_data = session_info.get('data', {})
state_dict = session_data.get('state_dict', {})

DEVOPS_FOLDER= get_devops_folder_name(DEVOPS_TOKEN)

#print(job_ids)
if api_trigger == "yes":
    print("Devops Observability for the inference job")

if inference_job_create_status:
    for job in list(job_ids.keys()):
        JOB_TYPE = job_ids[job].get("type")
        JOB_ID = job_ids[job].get("job_id")

        if job != inference_job_name:
            model_approval_id =  ""
            model_artifact_id  = ""
            status = ""
        else:
            status = "success"

        PAYLOAD = {
                "project_id": state_dict.get('project_id'),
                "project_name": state_dict.get('project_name'),
                "version": state_dict.get('version'),
                "job_id": f"{JOB_ID}",
                "job_type": JOB_TYPE,
                "git_hub_repo_url": GIT_REPO_URL,
                "databricks_repo_folder_name": DATABRICKS_REPO_FOLDER_NAME,
                "devops_organization_name": DEVOPS_ORG_NAME,
                "devops_project_name": DEVOPS_PROJECT_NAME,
                "pipeline_name": PIPELINE_NAME,
                "pipeline_definition_id": PIPELINE_ID,
                "build_id": PIPELINE_RUN_ID,
                "commit_id": COMMIT_ID,
                "commit_url": f"{GIT_REPO_URL}/commit/{COMMIT_ID}",
                "target_branch_name": TARGET_BRANCH,
                "build_url": f"{DEVOPS_ORG_URL}{DEVOPS_PROJECT_NAME}/_build/results?buildId={PIPELINE_RUN_ID}",
                "triggered_by": TRIGGERED_BY,
                "status" : status,
                "run_notebook_url":f"{DATABRICKS_HOST}/jobs/{JOB_ID}",
                "devops_orchestrator": "Azure Devops",
                "devops_badge" : f"{DEVOPS_ORG_URL}{DEVOPS_PROJECT_NAME}/_apis/build/status/{DEVOPS_FOLDER}/{PIPELINE_NAME}?branchName={TARGET_BRANCH}",
                "devops_version" : "v3",
                "model_artifact_id" : model_artifact_id,
                "model_approval_id" : model_approval_id
        }
    
        print(PAYLOAD)
        devops_data_add(API_BASE_URL,PAYLOAD, Header)

else:
    status = "failed"
    JOB_ID = ""
    JOB_TYPE = "Inference"
    



    PAYLOAD = {
                "project_id": state_dict.get('project_id'),
                "project_name": state_dict.get('project_name'),
                "version": state_dict.get('version'),
                "job_id": f"{JOB_ID}",
                "job_type": JOB_TYPE,
                "git_hub_repo_url": GIT_REPO_URL,
                "databricks_repo_folder_name": DATABRICKS_REPO_FOLDER_NAME,
                "devops_organization_name": DEVOPS_ORG_NAME,
                "devops_project_name": DEVOPS_PROJECT_NAME,
                "pipeline_name": PIPELINE_NAME,
                "pipeline_definition_id": PIPELINE_ID,
                "build_id": PIPELINE_RUN_ID,
                "commit_id": COMMIT_ID,
                "commit_url": f"{GIT_REPO_URL}/commit/{COMMIT_ID}",
                "target_branch_name": TARGET_BRANCH,
                "build_url": f"{DEVOPS_ORG_URL}{DEVOPS_PROJECT_NAME}/_build/results?buildId={PIPELINE_RUN_ID}",
                "triggered_by": TRIGGERED_BY,
                "status" : status,
                "run_notebook_url":f"{DATABRICKS_HOST}/jobs/{JOB_ID}",
                "devops_orchestrator": "Azure Devops",
                "devops_badge" : f"{DEVOPS_ORG_URL}{DEVOPS_PROJECT_NAME}/_apis/build/status/{DEVOPS_FOLDER}/{PIPELINE_NAME}?branchName={TARGET_BRANCH}",
                "devops_version" : "v3",
                "model_artifact_id" : model_artifact_id,
                "model_approval_id" : model_approval_id
        }
    
    print(PAYLOAD)
    devops_data_add(API_BASE_URL,PAYLOAD, Header)


#LOGIC FOR EXTERNAL JOB REGISTRATION OF META JOBS

for job in job_ids:
    print(job)
    if job_ids.get(job).get("type") in [TRAIN_META_JOB_TYPE , RETRAIN_META_JOB_TYPE]:
        print("hello")
        JOB_TYPE = job_ids[job].get("type")
        JOB_ID = job_ids[job].get("job_id")
        JOB_NAME =  job

        PAYLOAD = {
                "project_id": state_dict.get('project_id'),
                "version": state_dict.get('version'),
                "job_id": f"{JOB_ID}",
                "job_type": JOB_TYPE,
                "job_name": JOB_NAME,
                "external_reg":"yes",
                "env": ENV
        }
        print(PAYLOAD)
        register_ext_metajob(API_BASE_URL,PAYLOAD,Header)

