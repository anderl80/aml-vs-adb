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

from azureml.core import Environment, Model, Run
from azureml.core.webservice import AksWebservice
from azureml.core.compute import AksCompute
from azureml.core.model import InferenceConfig
from azureml.core.conda_dependencies import CondaDependencies

run = Run.get_context()
ws = run.experiment.workspace
service_name = "trump-tweets-scoring-aml"
model_name = "trump-tweet-classification"

# getting last model
model = next(iter(Model.list(workspace=ws, name=model_name, latest=True)), None,)

environment = Environment("trump-tweet-inferencing")
conda_dep = CondaDependencies()
conda_dep.add_conda_package("scikit-learn")
conda_dep.add_pip_package("azureml-sdk")
environment.python.conda_dependencies = conda_dep

aks_target = AksCompute(ws, "trump-tweets-inf")

deployment_config = AksWebservice.deploy_configuration(cpu_cores = 2, memory_gb = 8)

inference_config = InferenceConfig(entry_script="score.py",
                                   environment=environment)

service = Model.deploy(ws, service_name, [model], inference_config, deployment_config, aks_target, overwrite=True)
service.wait_for_deployment(show_output=True)