# Databricks notebook source
# Copyright (c) 2021, Microsoft

# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.

# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.
#
# DO NOT USE IN PRODUCTION ENVIRONMENTS.

# COMMAND ----------

# MAGIC %pip install azureml-sdk

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Deploy model for realtime inferencing on Azure ACI/AKS
# MAGIC 
# MAGIC AKS for production or MLFlow Model Serving for dev, too.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC TODO: pass run ID from model; but better would be get latest model from registry
# MAGIC 
# MAGIC https://docs.databricks.com/notebooks/notebook-workflows.html

# COMMAND ----------

import mlflow.azureml
from azureml.core import Workspace
from azureml.core.webservice import AksWebservice, AciWebservice, Webservice
from azureml.core.compute import AksCompute

# Load or create an Azure ML Workspace
workspace_name = "mlw-test"
subscription_id = dbutils.secrets.get(scope="key-vault-secrets", key="sandbox-subscription-id")
resource_group = "rg-ai-test"
location = "westeurope"

azure_workspace = Workspace.create(name=workspace_name,
                                   subscription_id=subscription_id,
                                   resource_group=resource_group,
                                   location=location,
                                   create_resource_group=True,
                                   exist_ok=True)

deploy_config = AksWebservice.deploy_configuration(cpu_cores = 2, memory_gb = 8, compute_target_name='trump-tweets-inf')

# Create an Azure Container Instance webservice for an MLflow model
azure_service, azure_model = mlflow.azureml.deploy(model_uri="runs:/{}/trump_tweets_pipe".format(run.info.run_id),
                                                   service_name="trump-tweets-scoring-adb",
                                                   deployment_config=deploy_config,
                                                   workspace=azure_workspace,
                                                   synchronous=True)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Test endpoints

# COMMAND ----------

import pandas as pd
sample_request = pd.DataFrame(["The media is spreading fake news!"], columns=["tweet"])

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Test ACI/AKS endpoint

# COMMAND ----------

import requests
import json
 
def query_endpoint_example(scoring_uri, inputs, service_key=None):
  headers = {
    "Content-Type": "application/json",
  }
  if service_key is not None:
    headers["Authorization"] = "Bearer {service_key}".format(service_key=service_key)
    
  print("Sending batch prediction request with inputs: {}".format(inputs))
  response = requests.post(scoring_uri, data=json.dumps(inputs), headers=headers)
  preds = json.loads(response.text)
  print("Received response: {}".format(preds))
  #return preds

query_endpoint_example("http://20.76.44.123:80/api/v1/service/trump-tweets-scoring-adb/score",
                       sample_request.to_dict(orient='split'),
                       dbutils.secrets.get(scope="key-vault-secrets", key="trump-tweets-scoring-aml"))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Test MLFLow model serving

# COMMAND ----------

pd.DataFrame.to_json(sample_request, orient='records')
