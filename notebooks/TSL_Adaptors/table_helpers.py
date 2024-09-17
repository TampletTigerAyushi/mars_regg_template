import requests
from requests.structures import CaseInsensitiveDict
import requests
from monitoring.utils.secret_mapping import SECRET_MAPPING
from monitoring.utils.vault_scope import VAULT_SCOPE

def get_app_url(dbutils):
    """
    Returns env and vault scope
    """
    print("Fetching API_ENDPOINT from secrets.")
    env, vault_scope = get_env_vault_scope(dbutils)
    API_ENDPOINT = ""
    if env in ["dev", "qa"]:
        API_ENDPOINT = (
            dbutils.secrets.get(
                scope=vault_scope,
                key=SECRET_MAPPING.get(f"az-app-service-{env}-url-2", ""),
            )
            + "/"
        )
    else:
        API_ENDPOINT = (
            dbutils.secrets.get(
                scope=vault_scope, key=SECRET_MAPPING.get(f"az-app-service-url", "")
            )
            + "/"
        )
    return API_ENDPOINT


def get_headers(vault_scope, dbutils):
    """
    Returns API headers
    """
    h1 = CaseInsensitiveDict()
    client_id = dbutils.secrets.get(
        scope=vault_scope, key=SECRET_MAPPING.get("az-api-client-id", "")
    )
    scope = client_id + "/.default"
    client_secret = dbutils.secrets.get(
        scope=vault_scope, key=SECRET_MAPPING.get("az-api-client-secret", "")
    )
    h1["Authorization"] = get_access_tokens(
        client_id, scope, client_secret, vault_scope, dbutils
    )
    h1["Content-Type"] = "application/json"
    return h1


def get_env_vault_scope(dbutils):
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
    print(f"ENV WITHIN TSL ADAPTOR IS {env}")
    if env == "prod":
        vault_scope = "qecdspdataprodqvc"
    elif env == "qa":
        vault_scope = "qecdspdataqaqvc"
    else:
        vault_scope = "qecdspdatadevqvc"
    try:
        if len(dbutils.fs.ls("dbfs:/FileStore/jars/MLCORE_INIT/vault_check.json")) == 1:
            if env == "qa":
                with open(
                    "/dbfs/FileStore/jars/MLCORE_INIT/vault_check_qa.json", "r"
                ) as file:
                    vault_check_data = json.loads(file.read())
            else:
                with open(
                    "/dbfs/FileStore/jars/MLCORE_INIT/vault_check.json", "r"
                ) as file:
                    vault_check_data = json.loads(file.read())
            if "@" in env:
                return "qa", vault_check_data["client_name"]
            return env, vault_check_data["client_name"]
        else:
            return env, vault_scope
    except:
        return env, vault_scope

def get_access_tokens(client_id, scope, client_secret, vault_scope, dbutils):
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
    tenant_id = dbutils.secrets.get(
        scope=vault_scope, key=SECRET_MAPPING.get("az-directory-tenant", "")
    )
    url = "https://login.microsoftonline.com/" + tenant_id + "/oauth2/v2.0/token"
    resp = requests.post(url, headers=headers, data=data).json()
    token = resp["access_token"]
    token_string = "Bearer" + " " + token
    return token_string
