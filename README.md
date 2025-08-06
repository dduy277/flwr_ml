#A Flower / PyTorch app

## Unzip the CSV file

The default csv file is df_2

## Install dependencies and project

In the project directory (ex:`flwr-sklearn-rnn`), use `pip install -e .` to install dependencies

```bash
pip install -e .
```

## Run with the Simulation Engine

## MLflow

To run with MLflow, start up the local MLflow server

```bash
mlflow server --host 127.0.0.1 --port 5000
```

In the project directory (ex:`flwr-sklearn-rnn`), use `flwr run` to run a local simulation:

```bash
flwr run .
```

If outside of the project directory, use `flwr run < path to project directory >` to run a local simulation:

```bash
flwr run flwr-sklearn-rnn
```