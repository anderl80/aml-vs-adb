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

import argparse
import numpy as np
import joblib
import json
import os
from azureml.core import Dataset, Run
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, f1_score, precision_score, recall_score, accuracy_score 

parser = argparse.ArgumentParser()
parser.add_argument("--input-data", type=str)
parser.add_argument("--min_doc_freq", type=int, default=5, dest='min_df')
parser.add_argument("--max_doc_freq", type=float, default=0.6, dest='max_df')
parser.add_argument("--ngram_min", type=int, default=4 , dest='ngram_min')
parser.add_argument("--ngram_max", type=int, default=7, dest='ngram_max')
parser.add_argument("--n_estimators", type=int, default=90, dest='n_estimators')
parser.add_argument("--max_depth", type=int, default=None, dest='max_depth')
parser.add_argument("--min_samples_split", type=int, default=2, dest='min_samples_split')
parser.add_argument("--min_class_frequency", type=int, default=50, dest='min_class_frequency')
args = parser.parse_args()
min_df = args.min_df
max_df = args.max_df
ngram_min = args.ngram_min
ngram_max = args.ngram_max
n_estimators = args.n_estimators
max_depth = args.max_depth
min_samples_split = args.min_samples_split
min_class_frequency = args.min_class_frequency
input_data = args.input_data

run = Run.get_context()
ws = run.experiment.workspace

# get the input dataset by ID
dataset = Dataset.get_by_id(ws, id=input_data)
run.log("Dataset Version", dataset.version)
data_ml = dataset.to_pandas_dataframe()

# take only data which KKS has more than 5 relating records
data_ml = data_ml.groupby('target').filter(lambda x: len(x) >= min_class_frequency)
# drop nulls
data_ml = data_ml[["target", "tweet"]].dropna().copy()

vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(ngram_min, ngram_max), min_df=min_df, max_df=max_df)

X = vec.fit_transform(data_ml['tweet'])

y = data_ml["target"]
run.log("Number of classes", len(set(y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_clf = RandomForestClassifier(n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                random_state=42)
rf_model = rf_clf.fit(X_train, y_train)

# predict
y_pred = rf_model.predict(X_test)

# calculate metrics
run.log('Accuracy', accuracy_score(y_test, y_pred))
run.log('Precision', precision_score(y_test, y_pred, average='weighted'))
run.log('Recall', recall_score(y_test, y_pred, average='weighted'))
run.log('F1', f1_score(y_test, y_pred, average='weighted'))

# Save the trained model in the outputs folder
os.makedirs('not_outputs/model', exist_ok=True)

with open('not_outputs/classification-report.json', 'wt') as of:
    of.write(json.dumps(classification_report(y_test, y_pred, output_dict=True)))

joblib.dump(value=rf_model, filename='not_outputs/model/trump-tweet-classification.pkl', compress=5)
joblib.dump(value=vec, filename='not_outputs/model/vec.pkl')
run.upload_folder('outputs', 'not_outputs')
