# Compare AML and ADB

This repository compares an end-to-end ML pipeline in Azure Machine Learning and Azure Databricks.

We'll use a text data classification ML approach to show how the pipelines could be set up in both services. In the end we'll deploy an endpoint using the model that was trained in both services.

## Deploy model

Depending on the use-case, the latency requirements might vary. Azure Databricks is capable of not only hosting, but also serving a model using MLFlow model serving. This means that a job cluster is spawned in the background to host the model. For dev/test workloads MLFlow serving or ACI is the place to go. Deploying to AKS is recommended in production workflows, especially when high-performance/low-latency is required. AKS is the best choice in terms of latency, is more scalable, can be fine tuned and has better cost control,

![Latencies](media/inferencing-latencies.png)