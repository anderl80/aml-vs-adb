# Compare AML and ADB

This repository compares an end-to-end ML pipeline in Azure Machine Learning and Azure Databricks.

We'll use a text data classification ML approach to show how the pipelines could be set up in both services. In the end we'll deploy an endpoint using the model that was trained in both services.

## Deploy model

Although Azure Databricks has MLFlow model serving (job cluster is spawn in background to host the model), deploying to AKS is recommended in production workflows, especially when high-performance/low-latency is required.