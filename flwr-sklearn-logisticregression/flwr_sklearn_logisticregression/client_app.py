"""flwr-sklearn-LogisticRegression: A Flower / sklearn app."""

import pandas as pd
import warnings
import json
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, log_loss, classification_report
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr_sklearn_logisticregression.task import (
    get_model,
    get_model_params,
    load_data,
    set_initial_params,
    set_model_params,
)
import mlflow
import mlflow.sklearn

class FlowerClient(NumPyClient):
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        #testing
    def fit(self, parameters, config):
        set_model_params(self.model, parameters)

        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)

        return get_model_params(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config): 
        set_model_params(self.model, parameters)
        
        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        # accuracy = self.model.score(self.X_test, self.y_test)
        precision, recall, thresholds = precision_recall_curve(self.y_test, y_score=self.model.predict_proba(self.X_test)[:, 1])
        ROC_AUC = roc_auc_score(self.y_test, self.model.predict_proba(self.X_test)[:, 1])
        AUC = auc(recall, precision)
        # print ("HERE: ",ROC_AUC)
        classification = classification_report(self.y_test, self.model.predict(self.X_test), target_names=['Not Fraud', 'Fraud'], output_dict=True)
        # Dict to json
        classification_str = json.dumps(classification)
        return loss, len(self.X_test), {"ROC_AUC": ROC_AUC, "AUC": AUC, "Classification_str": classification_str}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    X_train, X_test, y_train, y_test = load_data(partition_id, num_partitions)

    # Create LogisticRegression Model
    penalty = context.run_config["penalty"]
    local_epochs = context.run_config["local-epochs"]
    model = get_model(penalty, local_epochs)

    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(model)

    return FlowerClient(model, X_train, X_test, y_train, y_test).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
