#A Flower / PyTorch app

# Instruction 
You need to install [Python](https://www.python.org/downloads/)

You also need to create a python enviroment using the [default way](https://python.land/virtual-environments/virtualenv) but I recommend using [pyenv](https://github.com/pyenv/pyenv) or [pyenv-win](https://github.com/pyenv-win/pyenv-win) for Window.
## Unzip the CSV file

The default csv file is df_2

## Install dependencies and project

In the project directory (ex:`flwr-sklearn-LogisticRegression`), use `pip install -e .` to install dependencies

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
