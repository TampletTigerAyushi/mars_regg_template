import json
import os
import sys

import requests
from requests.structures import CaseInsensitiveDict

client_id = sys.argv[1]
client_secret = sys.argv[2]
tenant_id = sys.argv[3]

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
    return token

Databricks_Token = get_access_tokens(
    client_id, databricks_scope, client_secret, tenant_id
)

print(Databricks_Token)