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

# MAGIC %md
# MAGIC 
# MAGIC # Good to read
# MAGIC 
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

data = transform_tweet_data()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Write to Delta Table

# COMMAND ----------

data.write.format("delta").mode("overwrite").saveAsTable("trumptweets")

# COMMAND ----------


