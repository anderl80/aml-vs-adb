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

# MAGIC %pip install pandas==1.1.0 azureml-sdk

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Good to read:
# MAGIC 
# MAGIC * http://steventhornton.ca/blog/hyperparameter-tuning-with-hyperopt-in-python.html
# MAGIC * https://stackoverflow.com/questions/53579444/efficient-text-preprocessing-using-pyspark-clean-tokenize-stopwords-stemming

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Prepare data

# COMMAND ----------

from pyspark.ml.feature import StopWordsRemover
from nltk.stem.snowball import SnowballStemmer
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.functions import regexp_replace, col, split, udf

@udf(StringType())
def concat_strings(col):
  return " ".join(col)

def transform_tweet_data():
    # load
    data = spark.read.csv("dbfs:/FileStore/tweets/trump_insult_tweets_2014_to_2021.csv", header=True)
    # select
    data = data.select(split("tweet", " ").alias("tweet"), "target").dropna()
    # remove stopword
    remover = StopWordsRemover(inputCol='tweet', outputCol='tweet_clean')
    data = remover.transform(data)
    # stem
    stemmer = SnowballStemmer(language='english')
    stemmer_udf = udf(lambda tokens: [stemmer.stem(token) for token in tokens], ArrayType(StringType()))
    data = data.withColumn("tweet_stemmed", stemmer_udf("tweet_clean")).select('target', 'tweet_stemmed')
    # clean
    data = data.withColumn("tweet", regexp_replace(concat_strings("tweet_stemmed"), '"', "")).select("tweet", "target")
    
    return data

data_pd = transform_tweet_data().toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Split Dataframe in train and test

# COMMAND ----------

train = data_pd.groupby('target').sample(frac = 0.8)
test = data_pd.drop(train.index)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Hyperparameter optimization

# COMMAND ----------

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from hyperopt import hp, STATUS_OK
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import mlflow

search_space = {
  'min_class_frequency': hp.randint('min_class_frequency', 50, 100),
  'n_estimators': hp.randint('n_estimators', 5, 120),
  'min_df': hp.randint('min_df', 1, 10),
  'max_df': hp.uniform('max_df', 0.4, 0.8),
  'ngram_min': hp.randint('ngram_min', 2, 4),
  'ngram_max': hp.randint('ngram_max', 4, 10),
}

def objective(params):
    n_estimators = params["n_estimators"]
    min_df = params["min_df"]
    max_df = params["max_df"]
    ngram_min = params["ngram_min"]
    ngram_max = params["ngram_max"]
    min_class_frequency = params["min_class_frequency"]
    
    # https://stackoverflow.com/questions/30485151/python-pandas-exclude-rows-below-a-certain-frequency-count
    filtered = train.groupby('target').filter(lambda x: len(x) >= min_class_frequency)
    
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(ngram_min, ngram_max), min_df=min_df, max_df=max_df)
    X = vec.fit_transform(filtered['tweet'])

    y = filtered["target"]
    
    clf = RandomForestClassifier(n_estimators=n_estimators)
                                 
    accuracy=cross_val_score(clf, X, y, scoring="accuracy").mean()
    
    return {'loss': -accuracy, 'status': STATUS_OK}

# COMMAND ----------

from hyperopt import fmin, tpe, SparkTrials
import mlflow

mlflow.set_experiment("/Users/{}/demos/trump-tweets/Trump-tweets-insults".format(dbutils.secrets.get(scope="key-vault-secrets", key="adb-username")))
#https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/hyperopt-concepts.html#id3
with mlflow.start_run(run_name="RandomForestClassifier Hyperopt", nested=True):
    argmin = fmin(fn=objective,
                  space=search_space,
                  algo=tpe.suggest,
                  max_evals=18,
                  trials=SparkTrials(parallelism=6))

# COMMAND ----------

argmin

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Train the model using the best parameters

# COMMAND ----------

argmin = {'max_df': 0.6,
 'min_class_frequency': 50,
 'min_df': 5,
 'n_estimators': 100,
 'ngram_max': 7,
 'ngram_min': 3}

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Using sklearn's pipeline

# COMMAND ----------

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import mlflow

mlflow.set_experiment("/Users/{}/demos/trump-tweets/Trump-tweets-insults".format(dbutils.secrets.get(scope="key-vault-secrets", key="adb-username")))
with mlflow.start_run(run_name="RandomForestClassifier Best Parameter Pipeline") as run:
    n_estimators = argmin["n_estimators"]
    min_df = argmin["min_df"]
    max_df = argmin["max_df"]
    ngram_min = argmin["ngram_min"]
    ngram_max = argmin["ngram_max"]
    min_class_frequency = argmin["min_class_frequency"]

    filtered = train.groupby('target').filter(lambda x: len(x) >= min_class_frequency)
    X = filtered[['tweet']]
    y = filtered["target"]
    
    feature_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(ngram_min, ngram_max), min_df=min_df, max_df=max_df))
    ])

    feature_preprocessing_pipe = ColumnTransformer([
        ("features_preprocessor", feature_pipe, 'tweet')
    ])

    pipe = Pipeline([
      ('feature_preprocessor', feature_preprocessing_pipe),
      ('estimator', RandomForestClassifier(n_estimators=n_estimators))
    ])

    pipe.fit(X, y)
    mlflow.sklearn.log_model(pipe, "trump_tweets_pipe")

    test_filtered = test[test['target'].isin(filtered['target'].unique())]
    acc = pipe.score(test_filtered[['tweet']], test_filtered['target'])
    mlflow.log_metric('accuracy', acc)

# COMMAND ----------

data = ["The media is spreading fake news!"]
import pandas as pd
pipe.predict(pd.DataFrame(data, columns=['tweet']))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Use logged model

# COMMAND ----------

logged_model = 'runs:/{}/trump_tweets_pipe'.format(run.info.run_id)
loaded_model = mlflow.pyfunc.load_model(logged_model)
data = ["The media is spreading fake news!"]
import pandas as pd
loaded_model.predict(pd.DataFrame(data, columns=['tweet']))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Register model

# COMMAND ----------

model_uri = "runs:/{}/trump_tweets_pipe".format(run.info.run_id)
mv = mlflow.register_model(model_uri, "TrumpTweetsClassifier")
print("Name: {}".format(mv.name))
print("Version: {}".format(mv.version))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Deploy model for realtime inferencing on Azure ACI/AKS
# MAGIC 
# MAGIC AKS for production or MLFlow Model Serving for dev, too.

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

# COMMAND ----------


