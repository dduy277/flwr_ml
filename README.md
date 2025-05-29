#A Flower / PyTorch app

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

If outside of the project directory, use `flwr run <project directory name>` to run a local simulation:

```bash
flwr run flwr-sklearn-LogisticRegression
```

To run with MLflow, start up the local MLflow server

```bash
mlflow server --host 127.0.0.1 --port 5000
```
