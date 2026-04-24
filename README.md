# Introduction
The project aim to develop several credit card fraud detection models using Federated Learning with Homomorphic Encryption to determine the viability of collaborative machine training between multiple banks without exposing their clients private data, even to third-party servers.

It includes 4 models: GRU, RNN, LSTM and Multihead Attention. Each model has both Federated Learning and Federated Learning (outsize [fhe-flwr](https://github.com/dduy277/flwr_ml/tree/Split_models_base_on_dataset/Dataset_1(df_1)-models/fhe-flwr) folder) with Homomorphic Encryption (inside [fhe-flwr](https://github.com/dduy277/flwr_ml/tree/Split_models_base_on_dataset/Dataset_1(df_1)-models/fhe-flwr) folder) version.

Two datasets were preprocessed and used to compare different models locally in [Local_ml_models](https://github.com/dduy277/Local_ml_models).

Due to the two datasets having different features, it was split into two different model bundles (include both models version for df_1 and df_2) for ease of use.

# Instruction
## Prerequisite
Download all the files [here](https://github.com/dduy277/flwr_ml/archive/refs/heads/Split_models_base_on_dataset.zip) and unzip it.

You need to install Python, download the latest version [here](https://www.python.org/downloads/).

You also need to create a python environment (pyenv) with python version `3.12.8` using the [default way](https://python.land/virtual-environments/virtualenv) or using [pyenv](https://github.com/pyenv/pyenv) (recommend) for Linux or [pyenv-win](https://github.com/pyenv-win/pyenv-win) for Window..

Install the necessary files in “requirements.txt” using the following command insize the newly created python environment.
```bash
pip install -r /path/to/requirements.txt
```

## Chose a dataset and model to run
There are two models: df_1, df_2. Each is in its own folder (ex: `Dataset_1(df_1)-models`) and came with its own models and model parameters. You can choose either of them, but df_1 is much smaller and trains faster than df_2.

Unzip all of the zip files inside the CSV fodler.

There are 4 ml models that you can run for each dataset, each one is in a folder that starts with 'flwr-torch' (ex:`flwr-torch-lstm`), and those same models running with Homomorphic encryption inside `fhe-flwr` folder.

Choose a model and open a terminal window in that model directory.

## Install dependencies for the project
Actiave the pyenv (python environment) that you installed.

In the model directory (ex:`flwr-torch-lstm`), use `pip install -e .` to install all dependencies. You NEED to do this for the first time for each model, both Federated learning only and Federated learning with Homomorphic encryption models.
```bash
pip install -e .
```

## Run with only Federated learning
In the model directory you want to run (ex:`flwr-torch-lstm`), input this command in the terminal to activate `MLflow`, a web interface to monitor and save models.
```bash
mlflow server --host 127.0.0.1 --port 5000
```
Open a web browser and input http://127.0.0.1:5000 in the search bar to open the MLflow interface. Here you can view the model in training and are able to check the result of the finished model.

Open a new terminal in the model directory and use `flwr run .` to train the model.
```bash
flwr run .
```

## Run with Federated learning and Homomorphic encryption
*All 4 of the models that used Federated learning with Homomorphic encryption is located in `fhe-flwr` folder.
*These models will run much slower than their Federated learning only counterpart.

In the dataset directory (ex:`Dataset_1(df_1)-models`), input this command in the terminal to activate `MLflow`, a web interface to monitor and save models.
```bash
mlflow server --host 127.0.0.1 --port 5000
```
Open a web browser and input http://127.0.0.1:5000 in the search bar to open the MLflow interface. Here you can view the model in training and are able to check the result of the finished model.

Open a new terminal in the model you want to run directory (ex:`fhe-flwr/flwr-torch-lstm`) and use `flwr run .` to train the model.
```bash
flwr run .
```

### Workarround
If you get `ValueError: Cannot load file containing pickled data when allow_pickle=False` error, find `parameter.py` usually in `/home/<user>/.pyenv/versions/3.12.8/envs/<pyenv_name>/lib/python3.12/site-packages/flwr/common/parameter.py` and change `allow_pickle` in `ndarray_deserialized = np.load(bytes_io, allow_pickle=False)` to `allow_pickle=True`.

## Setting
You can change the model training parameter such as: epochs, training round,... under `[tool.flwr.app.config]` in `pyproject.toml`.

You can also change the Homomorphic encryption parameter in `fhe-flwr/flwr-torch-<model>/flwr_torch_<model>/Crypto/fhe_crypto.py` of each model.

## Deploying trained model
After training a model, it will apprear in MLflow interface. (Access with a web browser at http://127.0.0.1:5000)

Open a new terminal in the Dataset folder (ex:`Dataset_1(df_1)-models`) in pyenv and input `Deploy_model/Deploy_scenal_<test number you want to run>.py` from 1 - 3 to use the model in a scenario:

Scenario 1: Evaluate the models metric when detecting fraud with Precision, Recall, F1-score.

Scenario 2: Evaluate the models response time when detecting fraud.

Scenario 3: Measure both performance and system resources when receiving waves of detection requests, each one lasting 5 seconds.

```bash
python Deploy_model/Deploy_scenal_2.py
```

When `Input model name: ` appear , input a model name from MLflow (located in MLflow `Models` tab).

```bash
Gobal_flwr-fhe-torch-rnn_df1
```

## Other
### GPU
The models will use the CPU instead of the GPU due to Ray not supporting Intel Arc GPU at the time. You can use NVindia GPU by adding `options.backend.client-resources.num-gpus = 0.5` in `pyproject.toml`.

### Other message
The message: `UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.` in the few starting round is a normal process due to the model basically guessing without any or little training.

### Muti-head attention error
Federated Learning with Homomorphic Encryption using Multi-head Attention with df_2 will always measure 0 in all metrics, further research is needed to understand the performance degradation.

### Models_performance
<img width="906" height="1257" alt="Models_performance" src="https://github.com/user-attachments/assets/ea511523-dfb9-4012-a898-43d2d040ec61" />

### Project architecture
<img width="1276" height="1051" alt="Project architecture" src="https://github.com/user-attachments/assets/c6d67d92-0067-421f-90e5-814afdf0a6ee" />

# Core library
The machine learning and deep learning models were made using [Scikit-learn](https://scikit-learn.org/stable/index.html) and [Pytorch](https://github.com/pytorch/pytorch) respectively.

The Federated Learning framework used in this project is [flower](https://github.com/flwrlabs/flower).

The Homomorphic Encryption library that was used for encryption is [OpenFHE](https://openfhe.org/).

The record, monitor of metrics and models was done using [MLflow](https://github.com/mlflow/mlflow).
