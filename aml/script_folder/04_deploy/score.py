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

import json
import joblib
import numpy as np
from azureml.core.model import Model

# Called when the service is loaded
def init():
    global model
    global vec
    model_path = Model.get_model_path("trump-tweet-classification")
    model = joblib.load(model_path + '/trump-tweet-classification.pkl')
    vec = joblib.load(model_path + '/vec.pkl')

# Called when a request is received
def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        X = vec.transform([data[0]])
        res = json.dumps(dict(zip(model.classes_, model.predict_proba(X).flatten())))
    except:
        res = json.dumps({})
    return res
