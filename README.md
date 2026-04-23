# Introduction
The project aim to develop several credit card fraud detection models using Federated Learning with Homomorphic Encryption to determine the viability of collaborative machine training between multiple banks without exposing their clients private data, even to third-party servers.
This project include 5 models: Logistic Regression, GRU, RNN, LSTM and Multihead Attention.

# Instruction
## Prerequisite
Download all the files [here](https://github.com/dduy277/flwr_ml/archive/refs/heads/main.zip) and unzip it.

You need to install Python, download the latest version [here](https://www.python.org/downloads/).

You also need to create a python environment with python version `3.11.12` using the [default way](https://python.land/virtual-environments/virtualenv) but I recommend using [pyenv](https://github.com/pyenv/pyenv) for Linux or [pyenv-win](https://github.com/pyenv-win/pyenv-win) for Window.

## Unzip the CSV file
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
