import json
import os
import sys

import requests
from requests.structures import CaseInsensitiveDict

client_id = sys.argv[1]
client_secret = sys.argv[2]
tenant_id = sys.argv[3]
databricks_host = sys.argv[4]
parent_folder_path = sys.argv[5]
repo_url = sys.argv[6]
project_name = sys.argv[7]
branch = sys.argv[8]
group_name = sys.argv[9]

databricks_scope = "2ff814a6-3304-4ab8-85cb-cd0e6f879c1d/.default"


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
    url = "https://login.microsoftonline.com/" + tenant_id + "/oauth2/v2.0/token"
    resp = requests.post(url, headers=headers, data=data).json()
    token = resp["access_token"]
    token_string = "Bearer" + " " + token
    return token_string


def create_repo_directory(databricks_host, parent_folder_path, headers):
    try:
        url = f"{databricks_host}/api/2.0/workspace/mkdirs"
        data = {"path": parent_folder_path}
        resp = requests.post(url, headers=headers, json=data).json()
        print(resp)
        return resp
    except Exception as e:
        print(f"Exception :  {e}")


def add_repo_codebase(databricks_host, parent_folder_path, project_name, headers):
    try:
        url = f"{databricks_host}/api/2.0/repos"
        data = {
            "url": f"{repo_url}",
            "provider": "gitHub",
            "path": f"{parent_folder_path}/{project_name}",
        }
        resp = requests.post(url, headers=headers, json=data).json()
        print(resp)
        return resp
    except Exception as e:
        print(f"Exception :  {e}")

def get_repo_id(databricks_host, parent_folder_path, project_name, headers):
        try:
            url = f"{databricks_host}/api/2.0/repos?path_prefix={parent_folder_path}/{project_name}"
            resp = requests.get(url, headers=headers).json()
            print(f"get_repo id {resp}")
            resp_id = resp['repos'][0]['id']
            print(f"get_repo id {resp_id}")
            return resp_id
        except Exception as e:
            print(f"Exception :  {e}")


def sync_branch(databricks_host,repo_id, branch, headers):
    try:
        url = f"{databricks_host}/api/2.0/repos/{repo_id}"
        data = {"branch": f"{branch}"}
        resp = requests.patch(url, headers=headers, json=data).json()
        print(resp)
        return resp
    except Exception as e:
        print(f"Exception :  {e}")


def update_repo_permissions(databricks_host, repo_id, group_name, headers):
    try:
        url = f"{databricks_host}/api/2.0/permissions/repos/{repo_id}"
        data = {"access_control_list": 
                [
                    {
                        "group_name": f"{group_name}",
                        "permission_level": "CAN_MANAGE"
                    }
                ]
            }
        resp = requests.patch(url, headers=headers, json=data).json()
        print(resp)
        return resp
    except Exception as e:
        print(f"Exception :  {e}")


Databricks_Token = get_access_tokens(
    client_id, databricks_scope, client_secret, tenant_id
)

headers = {"Authorization": Databricks_Token}

create_repo_directory(databricks_host, parent_folder_path, headers)
add_repo_codebase(databricks_host, parent_folder_path, project_name, headers)
repo_id= get_repo_id(databricks_host, parent_folder_path, project_name, headers)
sync_branch(databricks_host,repo_id, branch, headers)
update_repo_permissions(databricks_host, repo_id, group_name, headers)

print(repo_id)