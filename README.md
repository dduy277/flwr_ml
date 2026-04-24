# Introduction
The project aim to develop several credit card fraud detection models using Federated Learning with Homomorphic Encryption to determine the viability of collaborative machine training between multiple banks without exposing their clients private data, even to third-party servers.

This project includes 4 models: GRU, RNN, LSTM and Multihead Attention. Each model has both Federated Learning and Federated Learning (outsize [fhe-flwr](https://github.com/dduy277/flwr_ml/tree/Split_models_base_on_dataset/Dataset_1(df_1)-models/fhe-flwr) folder) with Homomorphic Encryption (inside [fhe-flwr](https://github.com/dduy277/flwr_ml/tree/Split_models_base_on_dataset/Dataset_1(df_1)-models/fhe-flwr) folder) version.

Two datasets were preprocessed and used to compare different models locally in [Local_ml_models](https://github.com/dduy277/Local_ml_models).

Due to the two datasets having different features, it was split into two different model bundles (include both models version for df_1 and df_2) for ease of use.

# Instruction
## Prerequisite
Download all the files [here](https://github.com/dduy277/flwr_ml/archive/refs/heads/Split_models_base_on_dataset.zip) and unzip it.

You need to install Python, download the latest version [here](https://www.python.org/downloads/).

You also need to create a python environment (pyenv) with python version `3.11.12` using the [default way](https://python.land/virtual-environments/virtualenv) or using [pyenv](https://github.com/pyenv/pyenv) (recommend) for Linux or [pyenv-win](https://github.com/pyenv-win/pyenv-win) for Window..

### Unzip the CSV file
Unzip all of the .csv files inside the CSV folder

The default CSV file that the models use is df_2 (data frame 2).


## Chose a model
There are 5 models that you can run, each one is in a folder that starts with 'flwr-' (ex:`flwr-sklearn-LogisticRegression`).

Choose a model and open a cmd (Command Prompt) window in that model directory.

## Install dependencies and project
Actiave the pyenv (python environment) that you installed.

In the model directory (ex:`flwr-sklearn-LogisticRegression`), use `pip install -e .` to install all dependencies.

```bash
pip install -e .
```

## Run with the Simulation Engine
  
In the project directory (ex:`flwr-sklearn-LogisticRegression`), use `flwr run` to run a local simulation:

```bash
flwr run .
```

If outside of the project directory, use `flwr run < path to project directory >` to run a local simulation:

```bash
flwr run flwr-sklearn-LogisticRegression
```
## MLflow

To run with MLflow, start up the local MLflow server

```bash
mlflow server --host 127.0.0.1 --port 5000
```
