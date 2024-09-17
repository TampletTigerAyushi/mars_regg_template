import os
import sys
import time
import yaml
import requests
import json
from requests.structures import CaseInsensitiveDict
import yaml
from collections import defaultdict, deque

ENV = sys.argv[1]
TARGET_BRANCH = sys.argv[2]
DEVOPS_ORG_URL = sys.argv[3]
DATABRICKS_REPO_FOLDER_NAME = sys.argv[4]
DATABRICKS_HOST = sys.argv[5]
DATABRICKS_TOKEN = sys.argv[6]
USER_GROUP = sys.argv[7]


if ENV == "qa":
    DATABRICKS_REPO_FOLDER_NAME = f"{DATABRICKS_REPO_FOLDER_NAME}_QA"
    
DEVOPS_ORG_NAME = DEVOPS_ORG_URL.split("/")[-2]
TARGET_BRANCH = TARGET_BRANCH.replace("refs/heads/", "")

file_path = "azure-db-repos-pipelinesv3/utility/job_ids.json"

with open(file_path, "r") as json_file:
    job_ids = json.load(json_file)

def read_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def topological_sort(graph):
    # Step 1: Build adjacency list and indegree count
    adj_list = defaultdict(list)
    indegree = defaultdict(int)

    for node, data in graph.items():
        depends_on = data.get('depends_on')
        indegree[node] += 0  # Ensure all nodes are initialized in indegree
        if depends_on is None:
            continue
        if isinstance(depends_on, str):
            adj_list[depends_on].append(node)
            indegree[node] += 1
        else:
            for dependency in depends_on:
                adj_list[dependency].append(node)
                indegree[node] += 1

    # Step 2: Perform topological sort using Kahn's algorithm (BFS)
    topo_order = []
    zero_indegree_queue = [node for node in graph if indegree[node] == 0]
    print(zero_indegree_queue)
    while zero_indegree_queue:
        current = zero_indegree_queue.pop(0)
        topo_order.append(current)

        for neighbor in adj_list[current]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                zero_indegree_queue.append(neighbor)

    # Step 3: Check for cycles
    if len(topo_order) != len(graph):
        raise ValueError("Cycle detected. Topological sorting is not possible.")

    return topo_order


def generate_meta_template(sorted_jobs, jobs):
    orch_job_template = {
        'resources': {
            'jobs': [
                {
                    'train_meta_job': {
                        'name': 'train_meta_job',
                        'type' : 'Train_Meta_Job',
                        'tasks': [],
                        'access_control_list' : [{
                            'group_name' : USER_GROUP,
                            'permission_level' : 'CAN_MANAGE'}]
                    }
                }
            ],
            #'existing_cluster_id': '<existing_cluster_id>'  # Replace with your actual cluster ID
        }
    }
    tasks = orch_job_template['resources']['jobs'][0]['train_meta_job']['tasks']

    for job in sorted_jobs:
        task = {
            'task_key': job,
            'run_job_task': {
                'job_id': '<job_id placeholder>'
            }
        }
        dependencies = jobs[job].get('depends_on', [])
        if dependencies:
            task['depends_on'] = [{'task_key': dep} for dep in dependencies]
        
        tasks.append(task)
    return orch_job_template



current_directory = os.path.dirname(os.path.realpath(__file__))
target_path = os.path.abspath(os.path.join(current_directory, "..", ".."))
target_path = os.path.abspath(os.path.join(target_path, "job_config"))
yaml_file_path = os.path.abspath(os.path.join(target_path,'job_order_config.yaml'))

config = read_yaml_config(yaml_file_path)

# Get the jobs from the configuration
jobs = config['jobs']
print(jobs)
sorted_jobs = topological_sort(jobs)
print("Jobs in topologically sorted order:", sorted_jobs)

# Generate the new YAML file based on the sorted jobs
meta_job_template = generate_meta_template(sorted_jobs, jobs)

# Update job_ids
for job in meta_job_template['resources']['jobs']:
    for task in job['train_meta_job']['tasks']:
        task_key = task.get('task_key')
        if task_key in job_ids:
            task['run_job_task']['job_id'] = int(job_ids[task_key].get("job_id"))


json_string =  json.dumps(meta_job_template['resources']['jobs'][0]['train_meta_job'])
payload =  json.loads(json_string)
print(payload)
databricks_url  = f"{DATABRICKS_HOST}"
databricks_token  =  f"{DATABRICKS_TOKEN}"
create_job_url = f"{databricks_url}/api/2.1/jobs/create"
headers = {
    'Authorization': f'Bearer {databricks_token}',
    'Content-Type': 'application/json'
}
try:
    response = requests.post(create_job_url, headers=headers,json=payload)
    result = response.json()
    print(result)
    print(f"Meta_job_id : {result.get('job_id')}")
except Exception as e:
    print(e)

for job in job_ids:
    if job not in sorted_jobs:
        print(f"{job} job_id : {job_ids.get(job)}")

job_dict_object= {"type" : meta_job_template['resources']['jobs'][0]['train_meta_job'].get('type'),
                          "job_id" : result.get('job_id')}
job_ids[meta_job_template['resources']['jobs'][0]['train_meta_job'].get('name')] =job_dict_object
with open(file_path, "w") as json_file:
    json.dump(job_ids, json_file, indent=4)