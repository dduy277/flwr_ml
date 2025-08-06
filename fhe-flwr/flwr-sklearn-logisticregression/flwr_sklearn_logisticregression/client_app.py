"""flwr-sklearn-LogisticRegression: A Flower / sklearn app."""

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
from flwr_sklearn_logisticregression.Crypto.fhe_crypto import FheCryptoAPI
import pickle
import numpy as np


class FlowerClient(NumPyClient):
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


    def __decrypt_params_if_needed(self, parameters, config):
        """Decrypt parameters if they are encrypted, otherwise return as-is"""
        if config.get('skip', False):
            return parameters
            
        cc = config.get('crypto_context')
        seckey = config.get('secret_key')
        
        if cc and seckey:
            # Parameters are encrypted, decrypt them
            decrypted_params = []
            for i, param_data in enumerate(parameters):
                if isinstance(param_data, list):
                    # encrypted_blocks = pickle.loads(param_data)
                    # Handle case where param_data is already a list of encrypted blocks
                    if i == 0:  # coef_ parameter
                        expected_shape = (self.model.classes_.shape[0], self.X_train.shape[1])
                    else :  #i == 1 intercept_ parameter (fit_intercept=True) 
                        expected_shape = (self.model.classes_.shape[0],)
                    decrypted_array = FheCryptoAPI.decrypt_numpy_array(
                        cc, seckey, param_data, np.float64, expected_shape
                    )
                    decrypted_params.append(decrypted_array)
                    print(f"Parameter IS encrypted")
                else:
                    # Parameters are already numpy arrays (not encrypted)
                    decrypted_params.append(param_data)
                    print(f"Parameter NOT encrypted")
            return decrypted_params
        else:
            # Parameters are not encrypted
            print("Parameters NOT encrypted")
            return parameters

        
    def __encrypt_params(self, parameters, config):
        """Encrypt parameters if crypto context is available"""
        if config.get('skip', False):
            return parameters
            
        cc = config.get('crypto_context')
        pubkey = config.get('public_key')

        # Encrypt the parameters
        encrypted_params = []
        for param in parameters:
            encrypted_blocks = FheCryptoAPI.encrypt_numpy_array(cc, pubkey, param)
            encrypted_params.append(pickle.dumps(encrypted_blocks))
        return encrypted_params


    def fit(self, parameters, config):
        # Decrypt parameters
        decrypted_params = self.__decrypt_params_if_needed(parameters, config)
        # print(decrypted_params)
        set_model_params(self.model, decrypted_params)

        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)

        updated_weights = get_model_params(self.model)
        encrypted_weights = self.__encrypt_params(updated_weights, config)
        return encrypted_weights, len(self.X_train), {}


    def evaluate(self, parameters, config):
        # Decrypt parameters if needed
        decrypted_params = self.__decrypt_params_if_needed(parameters, config)
        set_model_params(self.model, decrypted_params)       
        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        # accuracy = self.model.score(self.X_test, self.y_test)
        precision, recall, thresholds = precision_recall_curve(self.y_test, y_score=self.model.predict_proba(self.X_test)[:, 1])
        ROC_AUC = roc_auc_score(self.y_test, self.model.predict_proba(self.X_test)[:, 1])
        AUC = auc(recall, precision)
        classification = classification_report(self.y_test, self.model.predict(self.X_test), target_names=['Not Fraud', 'Fraud'], output_dict=True)
        # Dict to json
        classification_str = json.dumps(classification)
        return loss, len(self.X_test), {"ROC_AUC": ROC_AUC, "AUC": AUC, "Classification_str": classification_str, "Loss": loss}


def client_fn(context: Context):
    # Load model and data
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
