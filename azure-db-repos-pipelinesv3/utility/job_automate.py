import yaml
import requests
import json
import os 
import sys
from requests.structures import CaseInsensitiveDict

project_name = sys.argv[1]
folder_name = sys.argv[2]
databricks_url = sys.argv[3]
databricks_token = sys.argv[4]
env = sys.argv[5]
API_BASE_URL = sys.argv[6]
AZ_CLIENT_ID = sys.argv[7]
AZ_CLIENT_SECRET = sys.argv[8]
AZ_TENANT = sys.argv[9]
PIPELINE_ID = sys.argv[10]
USER_GROUP = sys.argv[11]
PIPELINE_RUN_ID = sys.argv[12]
PIPELINE_NAME = sys.argv[13]
DEVOPS_ORG_URL = sys.argv[14]
DEVOPS_PROJECT_NAME = sys.argv[15]
model_name =  sys.argv[16]
model_version = sys.argv[17]
version_env =  sys.argv[18]
api_trigger = sys.argv[19]
model_approval_id =  sys.argv[20]
model_artifact_id  = sys.argv[21]

DEVOPS_ORG_NAME = DEVOPS_ORG_URL.split("/")[-2]  
print(f"Pipelinedefinition ID: {PIPELINE_ID}")

if env == "qa":
    folder_name = f"{folder_name}_QA"

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

def check_if_exists(API_BASE_URL,Header, job_type, pipeline_definition_id):
    # Define the URL
    url = f"https://mlcoredevv2pg21.azurewebsites.net/mlapi/devops/data/get?pipeline_definition_id={pipeline_definition_id}&job_type={job_type}"
    
    # Define the Bearer token
    token = Header

    # Set up the headers with the Bearer token
    headers = {
        "Authorization": f"Bearer {token}"
    }
    # Send the GET request
    #print(token)
    response = requests.get(url,headers=token)

    return response
    
Header = get_headers(AZ_CLIENT_ID, AZ_CLIENT_SECRET, AZ_TENANT)




def read_yaml_files(directory):
    yaml_data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                yaml_content = yaml.safe_load(file)
                yaml_data[filename] = yaml_content
    return yaml_data


def create_JSON(jobs_yaml_directory = "../../resources"):
    """
    Returns: 
    dic : 
    {"job_name" :  "payload" }
    """
    # Example usage:
    yaml_dict = read_yaml_files(jobs_yaml_directory)
    payloads = {}
    
    for file,elems in yaml_dict.items():
        for job_key, job in elems['resources'].get('jobs').items():
        
            payloads[job_key] =  job
    
    return payloads


#print(create_JSON())
current_directory = os.path.dirname(os.path.realpath(__file__))
target_path = os.path.abspath(os.path.join(current_directory, "..", ".."))
target_path = os.path.abspath(os.path.join(target_path, "resources"))

#print(create_JSON(jobs_yaml_directory = target_path))


job_config_dict = create_JSON(jobs_yaml_directory = target_path)
job_list = list(job_config_dict.keys())
# print(job_list)
# print(job_config_dict)

if os.path.exists(f"data_config/SolutionConfig_{env}.yaml"):
    yaml_file_path = f"data_config/SolutionConfig_{env}.yaml"
else:
    yaml_file_path = "data_config/SolutionConfig.yaml"
with open(yaml_file_path, 'r') as file:
    yaml_content = file.read()
solution_config = yaml.safe_load(yaml_content)


create_job_url = f"{databricks_url}/api/2.1/jobs/create"
headers = {
    'Authorization': f'Bearer {databricks_token}',
    'Content-Type': 'application/json'
}

databricks_yml_path = "databricks.yml"
with open(databricks_yml_path, 'r') as file:
    yaml_content = file.read()
databricks_config = yaml.safe_load(yaml_content)

job_cluster_key = databricks_config['variables'].get('job_cluster_key')
libraries = databricks_config['variables'].get('libraries')
job_clusters = databricks_config['variables'].get('job_clusters')
job_ids = {}

print(job_list)


if api_trigger == "yes":
    inference_job_name =  model_name+"_"+"inference"
    print(solution_config)
    print(solution_config["inference"])
    solution_config["inference"]["datalake_configs"]["output_tables"]["output_1"]["table"] = solution_config["inference"]["datalake_configs"]["output_tables"]["output_1"]["table"] + "_" + model_name + "_" + model_version + "_" + version_env 
    if inference_job_name not in job_list: 
        print("Mismatch ")
        sys.exit()

for job in job_list:

    if api_trigger == "yes":
        if inference_job_name != job:
            continue

    if job_config_dict[job].get('type') == "Inference" and api_trigger=="no":
        print("Skipping inference creration")
        continue
    
    #This if block is deprecated
    # if job_config_dict[job].get("is_deterministic",False):
    #     continue
    
    # response =  check_if_exists(API_BASE_URL,Header,job,PIPELINE_ID)
    # response_dict = json.loads(response.content.decode('utf-8'))
    # if(len(response_dict.get("data"))>0):
    #     print(f"{job} already exists")
    #     job_ids[job] = response_dict.get("data")[0]["job_id"]
    #     continue

    job_config_dict[job]["job_clusters"] =  job_clusters
    job_config_dict[job]["access_control_list"] = [{'group_name' : USER_GROUP,
                                                'permission_level' : 'CAN_MANAGE'}]

    for task in job_config_dict[job]['tasks']:

        #Iterate through each task in the job_config_dict

        #update the notebook path as the per databricks repo
        task["notebook_task"]["notebook_path"]=f'/Repos/{folder_name}/{project_name}{task["notebook_task"]["notebook_path"]}'
        #print(task["notebook_task"]["notebook_path"])
        
        #adding solution config content as the job param
        task["notebook_task"]["base_parameters"]["env"] =  env
        task["notebook_task"]["base_parameters"]["sdk_session_id"] =  solution_config.get("general_configs").get("sdk_session_id").get(env)
        task["notebook_task"]["base_parameters"]["solution_config"] = json.dumps(solution_config)
        task["notebook_task"]["base_parameters"]["pipeline_id"] = PIPELINE_ID
        task["notebook_task"]["base_parameters"]["build_id"] = PIPELINE_RUN_ID
        task["notebook_task"]["base_parameters"]["pipeline_name"] = PIPELINE_NAME
        task["notebook_task"]["base_parameters"]["pipeline_type"] = 'azdevops'
        task["notebook_task"]["base_parameters"]["organization"] = DEVOPS_ORG_NAME
        task["notebook_task"]["base_parameters"]["project"] = DEVOPS_PROJECT_NAME
        if api_trigger == "yes":
            task["notebook_task"]["base_parameters"]["model_version"] = model_version
        
        
        #Pre-requisite libarary installation in the job_cluster 
        #task["libraries"] = libraries

        #Assign the job_cluster key (SHOULD BE ASSIGNED DYNAMICALLY IN FUTURE)
        #task["job_cluster_key"] = job_cluster_key
        task["existing_cluster_id"] = "0614-142133-xvsblpkr"

        #print(job_config_dict[job])
        
    PAYLOAD = job_config_dict[job]
    print(PAYLOAD)
    
    try:
        response = requests.post(create_job_url, headers=headers,json=PAYLOAD)
        result = response.json()
        print(f"{job} : {result}")
        job_dict_object= {"type" : job_config_dict[job]['type'],
                          "job_id" : result["job_id"]}

        job_ids[job] = job_dict_object
    except Exception as e:
        print(e)

#Structure of job_ids ==> {"job_name" : {"type" : <type> ,  "job_id" : <val> }}
print(job_ids)

file_path =  "azure-db-repos-pipelinesv3/utility/job_ids.json"
with open(file_path, "w") as json_file:
    json.dump(job_ids, json_file, indent=4)
