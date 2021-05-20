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

# MAGIC %pip install pandas==1.2.4

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Good to read
# MAGIC 
# MAGIC * http://steventhornton.ca/blog/hyperparameter-tuning-with-hyperopt-in-python.html

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Read from Delta Table

# COMMAND ----------

data = spark.read.table("trumptweets").toPandas() 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Split Dataframe in train and test

# COMMAND ----------

train = data.groupby('target').sample(frac = 0.8, random_state=42)
test = data.drop(train.index)

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
                  max_evals=6,
                  trials=SparkTrials(parallelism=6))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Train the model using the best parameters

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

# MAGIC %md
# MAGIC 
# MAGIC # Use logged model

# COMMAND ----------

#data = ["The media is spreading fake news!"]
#import pandas as pd
#pipe.predict(pd.DataFrame(data, columns=['tweet']))

#logged_model = 'runs:/{}/trump_tweets_pipe'.format(run.info.run_id)
#loaded_model = mlflow.pyfunc.load_model(logged_model)
#data = ["The media is spreading fake news!"]
#import pandas as pd
#loaded_model.predict(pd.DataFrame(data, columns=['tweet']))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Register model

# COMMAND ----------

model_uri = "runs:/{}/trump_tweets_pipe".format(run.info.run_id)
mv = mlflow.register_model(model_uri, "TrumpTweetsClassifier")
print("Name: {}".format(mv.name))
print("Version: {}".format(mv.version))
