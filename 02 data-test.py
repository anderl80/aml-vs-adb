# Databricks notebook source
# MAGIC %md
# MAGIC # Useful things how to set up in hosted environments
# MAGIC 
# MAGIC - https://youtu.be/pq5CBea12v4
# MAGIC - https://docs.greatexpectations.io/en/latest/guides/how_to_guides/configuring_data_contexts/how_to_instantiate_a_data_context_on_a_databricks_spark_cluster.html
# MAGIC - https://docs.greatexpectations.io/en/latest/guides/how_to_guides/configuring_data_contexts/how_to_instantiate_a_data_context_without_a_yml_file.html

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Context (Project setup)

# COMMAND ----------

from great_expectations.data_context.types.base import DataContextConfig, DatasourceConfig, FilesystemStoreBackendDefaults
from great_expectations.data_context.store.tuple_store_backend import TupleAzureBlobStoreBackend
from great_expectations.data_context import BaseDataContext

AZURE_STORAGE_CONNECTION_STRING = dbutils.secrets.get(scope="key-vault-secrets", key="gesuite-blob-storage-constring")

data_context_config = DataContextConfig(
    datasources={"dbw_test_datasource": DatasourceConfig(
        class_name="SparkDFDatasource",
        batch_kwargs_generators={
            "subdir_reader": {
                "class_name": "SubdirReaderBatchKwargsGenerator",
                "base_directory": "/FileStore/tweets/",
            }
        },
    )
               },
    stores={
          "expectations_store": {
              "class_name": "ExpectationsStore",
              "store_backend": {
                  "class_name": "TupleAzureBlobStoreBackend",
                  "container":  "gesuite-trumptweets",
                  "connection_string": AZURE_STORAGE_CONNECTION_STRING,
                  "prefix": "expectations"
              },
          },
          "validations_store": {
              "class_name": "ValidationsStore",
              "store_backend": {
                  "class_name": "TupleAzureBlobStoreBackend",
                  "container":  "gesuite-trumptweets",
                  "connection_string": AZURE_STORAGE_CONNECTION_STRING,
                  "prefix": "validations"
              },
          },
          "evaluation_parameter_store": {
              "class_name": "EvaluationParameterStore"
          },
          "checkpoint_store": {
              "class_name": "CheckpointStore",
              "store_backend": {
                  "class_name": "TupleAzureBlobStoreBackend",
                  "container": "gesuite-trumptweets",
                  "prefix": "checkpoint",
                  "connection_string": AZURE_STORAGE_CONNECTION_STRING
            },
        },
    },
    expectations_store_name="expectations_store",
    validations_store_name="validations_store",
    checkpoint_store_name="checkpoint_store",
    evaluation_parameter_store_name="evaluation_parameter_store",
    data_docs_sites = {
        "az_site": {
            "class_name": "SiteBuilder",
            "store_backend": {
                "class_name": "TupleAzureBlobStoreBackend",
                "container":  "\$web",
                "connection_string": AZURE_STORAGE_CONNECTION_STRING
            },
            "site_index_builder": {
                "class_name": "DefaultSiteIndexBuilder",
            },
        }
    },
    validation_operators={
        "action_list_operator": {
            "class_name": "ActionListValidationOperator",
            "action_list": [
                {
                    "name": "store_validation_result",
                    "action": {"class_name": "StoreValidationResultAction"},
                },
                {
                    "name": "store_evaluation_params",
                    "action": {"class_name": "StoreEvaluationParametersAction"},
                },
                {
                    "name": "update_data_docs",
                    "action": {"class_name": "UpdateDataDocsAction"},
                }
            ],
        }
    },
    store_backend_defaults = TupleAzureBlobStoreBackend(container="gesuite-trumptweets", connection_string=AZURE_STORAGE_CONNECTION_STRING)
    #store_backend_defaults = FilesystemStoreBackendDefaults(root_directory="/dbfs/FileStore/ge-trumptweets"),
)
context = BaseDataContext(project_config=data_context_config)

# COMMAND ----------

import json

print(json.dumps(context.list_datasources()[0], indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC # Create batch

# COMMAND ----------

from great_expectations.data_context import BaseDataContext

file_location = "/FileStore/tweets/trump_insult_tweets_2014_to_2021.csv"
file_type = "csv"
infer_schema = "true"
first_row_is_header = "true"

df = spark.read.format(file_type) \
    .option("inferSchema", infer_schema) \
    .option("header", first_row_is_header) \
    .load(file_location)

context.create_expectation_suite("trump-tweets-ge-test", overwrite_existing=True)

my_batch = context.get_batch({
    "dataset": df,
    "datasource": "dbw_test_datasource",
}, "trump-tweets-ge-test")

# COMMAND ----------

my_batch.head()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Profile data

# COMMAND ----------

import great_expectations as ge
from great_expectations.profile.basic_dataset_profiler import BasicDatasetProfiler

gf = ge.from_pandas(df.toPandas())
expectation_suite, validation_result = BasicDatasetProfiler.profile(gf)
#context.save_expectation_suite(expectation_suite, "my_profiled_expectations")

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Expectations

# COMMAND ----------

my_batch.expect_column_values_to_not_match_regex("tweet", '\"', mostly=0.95, meta={"notes": "I don't like quotations marks."})

# COMMAND ----------

my_batch.expect_column_values_to_not_be_null("tweet", mostly=0.95)

# COMMAND ----------

# MAGIC %md
# MAGIC # Save Expectation Suite

# COMMAND ----------

my_batch.save_expectation_suite(discard_failed_expectations=True)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Validate batch

# COMMAND ----------

val_batch = context.get_batch({
    "dataset": df,
    "datasource": "dbw_test_datasource"
}, "trump-tweets-ge-test")

# COMMAND ----------

import datetime

run_id = {
  "run_name": "Test validation run",
  "run_time": datetime.datetime.now(datetime.timezone.utc)
}

results = context.run_validation_operator(
    "action_list_operator",
    assets_to_validate=[val_batch],
    run_id=run_id)

# COMMAND ----------

context.build_data_docs()

# COMMAND ----------


