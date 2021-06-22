# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC https://youtu.be/pq5CBea12v4
# MAGIC https://docs.greatexpectations.io/en/latest/guides/how_to_guides/configuring_data_contexts/how_to_instantiate_a_data_context_on_a_databricks_spark_cluster.html

# COMMAND ----------

from great_expectations.data_context.types.base import DataContextConfig, DatasourceConfig, FilesystemStoreBackendDefaults
from great_expectations.data_context import BaseDataContext

# Example filesystem Datasource
my_spark_datasource_config = DatasourceConfig(
    class_name="SparkDFDatasource",
    batch_kwargs_generators={
        "subdir_reader": {
            "class_name": "SubdirReaderBatchKwargsGenerator",
            "base_directory": "/FileStore/tables/",
        }
    },
)

data_context_config = DataContextConfig(
    datasources={"dbw_test_datasource": my_spark_datasource_config},
    store_backend_defaults=FilesystemStoreBackendDefaults(root_directory="/dbfs/FileStore/"),
)
context = BaseDataContext(project_config=data_context_config)

# COMMAND ----------

context.list_datasources()

# COMMAND ----------

from great_expectations.data_context import BaseDataContext

file_location = "/FileStore/tweets/trump_insult_tweets_2014_to_2021.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
    .option("inferSchema", infer_schema) \
    .option("header", first_row_is_header) \
    .load(file_location)

# NOTE: project_config is a DataContextConfig set up as in the examples above.
context.create_expectation_suite("trump-tweets-ge-test", overwrite_existing=True)

my_batch = context.get_batch({
    "dataset": df,
    "datasource": "dbw_test_datasource",
}, "trump-tweets-ge-test")

# COMMAND ----------

my_batch.head()

# COMMAND ----------

my_batch.expect_column_values_to_not_match_regex("tweet", '\"', mostly=0.95)

# COMMAND ----------

my_batch.save_expectation_suite(discard_failed_expectations=False)

# COMMAND ----------


