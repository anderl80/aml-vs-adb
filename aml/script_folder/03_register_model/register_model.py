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

from azureml.core import Run
from azureml.pipeline.steps import HyperDriveStepRun
from azureml.core import Workspace
from azureml.pipeline.core.run import PipelineRun


run = Run.get_context()
pipeline_run = PipelineRun(run.experiment, run.parent.id)
train_models_step_run = HyperDriveStepRun(step_run=pipeline_run.find_step_run("hyperdrive_step")[0])
best_run = train_models_step_run.get_best_run_by_primary_metric()
final_test_accuracy = best_run.get_metrics("Accuracy")["Accuracy"]

best_run.register_model(
    model_name="trump-tweet-classification", model_path="outputs/model",
    tags={"Final Test Accuracy": str(final_test_accuracy), 'label': 'target'},
    properties={'Accuracy': best_run.get_metrics()['Accuracy']}
)