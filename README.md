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

Install the necessary files in “requirements.txt” using the following command insize the newly created python environment.
```bash
pip install -r /path/to/requirements.txt
```

## Chose a dataset and model to run
There are two models: df_1, df_2. Each is in its own folder (ex: `Dataset_1(df_1)-models`) and came with its own models and model parameters. You can choose either of them, but df_1 is much smaller and trains faster than df_2.

Unzip all of the zip files inside the CSV fodler.

There are 5 models that you can run for each dataset, each one is in a folder that starts with 'flwr-torch' (ex:`flwr-torch-lstm`).

Choose a model and open a terminal window in that model directory.

## Install dependencies for the project
Actiave the pyenv (python environment) that you installed.

In the model directory (ex:`flwr-torch-lstm`), use `pip install -e .` to install all dependencies.
```bash
pip install -e .
```

## Run with only Federated learning
In the model directory (ex:`flwr-torch-lstm`), input this command in the terminal to activate `MLflow`, a web interface to monitor and save models.
```bash
mlflow server --host 127.0.0.1 --port 5000
```
Open a web browser and input http://127.0.0.1:5000 in the search bar to open the MLflow interface. Here you can view the model in training and are able to check the result of the finished model.

Open a new terminal in the model directory and use `flwr run .` to train the model.
```bash
flwr run .
```

If outside of the model directory, use `flwr run < path to project directory >` to run a local simulation, eg:
```bash
flwr run flwr-torch-lstm
```
