"""flwr-sklearn-LogisticRegression: A Flower / sklearn app."""

import numpy as np
import pandas as pd
from flwr_datasets.partitioner import IidPartitioner
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from datasets import Dataset
import mlflow


def load_data(partition_id: int, num_partitions: int):
    """Load partition df_3 data."""
    df = pd.read_csv('CSV/df_train_3.csv')
    df.drop("Unnamed: 0", axis=1, inplace=True)
    dataset = Dataset.from_pandas(df)
    partitioner = IidPartitioner(num_partitions=num_partitions)
    # partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by="Class", alpha=10, min_partition_size=5)
    partitioner.dataset = dataset
    dataset = partitioner.load_partition(partition_id=partition_id).to_pandas()

    # ".values" to fix: X has feature names, but LogisticRegression was fitted without feature names
    X = dataset.drop('Class', axis=1).values
    y = dataset['Class'].values
    # Split the on edge data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test
    

def get_model(penalty: str, local_epochs: int):
    params = {
        "penalty":penalty,
        "max_iter":local_epochs,
        "warm_start":False,
    }
    return LogisticRegression(**params)

# get local parameters that has been updated (trained)
def get_model_params(model):
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [model.coef_]
    return params


def set_model_params(model, params):
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

# get initial parameters
def set_initial_params(model):
    n_classes = 2  # Number of class in dataset (y)
    n_features = 16  # Number of features in dataset
    model.classes_ = np.array([i for i in range(n_classes)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))
