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

from genericpath import exists
from azureml.core import Dataset, Run
import argparse
import pandas as pd
import os

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
args = parser.parse_args()
print("input argument: %s" % args.input)

run = Run.get_context()
ws = run.experiment.workspace

data = pd.read_csv(args.input + '/trump_insult_tweets_2014_to_2021.csv', sep=',', header='infer')
data = data[["target", "tweet"]].dropna()

# export file
os.makedirs("output", exist_ok=True)
data.to_csv("output/prepared.csv", index=False)

# get the datastore to upload prepared data
datastore = ws.get_default_datastore()

# upload the local file from src_dir to the target_path in datastore
datastore.upload(src_dir='output', target_path='data', overwrite=True)

# create a dataset referencing the cloud location
dataset = Dataset.Tabular.from_delimited_files(path=[(datastore, ('data/prepared.csv'))])

dataset = dataset.register(workspace=ws,
                           name='trump-tweets-prepared',
                           create_new_version=True)