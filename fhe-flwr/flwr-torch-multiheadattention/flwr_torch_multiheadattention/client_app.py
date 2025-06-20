"""flwr-torch-MultiheadAttention: A Flower / PyTorch app."""

import torch
import pickle

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr_torch_multiheadattention.task import Net, get_weights, load_data, set_weights, test, train
from flwr_torch_multiheadattention.Crypto.fhe_crypto import FheCryptoAPI
import json
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, classification_report



## Hyper-parameters 
input_dim = 1
dim_model = 64
num_classes = 2 # num y class
num_heads = 4

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        if torch.xpu.is_available():    # for Intel GPU
            self.device = torch.device("xpu:0")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)


    def __decrypt_params_if_needed(self, parameters, config):
        """Decrypt parameters if they are encrypted, otherwise return as-is"""
        if config.get('skip', False):
            return parameters
            
        cc = config.get('crypto_context')
        seckey = config.get('secret_key')
        
        if cc and seckey:
            # Parameters are encrypted, decrypt them
            decrypted_params = []
            for i, (param_data, model_param) in enumerate(zip(parameters, self.net.state_dict().values())):
                # Decrypted parameters blocks directly
                if isinstance(param_data, list):
                    decrypted_tensor = FheCryptoAPI.decrypt_torch_tensor(
                        cc, seckey, param_data, 
                        model_param.dtype, model_param.shape
                    )
                    decrypted_params.append(decrypted_tensor.cpu().numpy())
                else:
                    # Parameters are already numpy arrays (not encrypted)
                    decrypted_params.append(param_data)
                    print("Parameters NOT encrypted")
            return decrypted_params
        else:
            # Parameters are not encrypted
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
        set_weights(self.net, decrypted_params)
        train_loss = train(
            net=self.net,
            trainloader=self.trainloader,
            epochs=self.local_epochs,
            device=self.device,
        )
        # Get updated weights and encrypt it
        updated_weights = get_weights(self.net) 
        encrypted_weights = self.__encrypt_params(updated_weights, config)

        return (
            encrypted_weights,
            len(self.trainloader),
            {"train_loss": train_loss},
        )


    def evaluate(self, parameters, config):
        # Decrypt parameters if needed
        decrypted_params = self.__decrypt_params_if_needed(parameters, config)
        set_weights(self.net, decrypted_params)
        loss, accuracy, X_preds, y_labels = test(self.net, self.valloader, self.device)
        # Precision-Recall curve and ROC-AUC score
        precision, recall, thresholds = precision_recall_curve(y_labels, X_preds)
        ROC_AUC = roc_auc_score(y_labels, X_preds)
        AUC = auc(recall, precision)
        # Convert probabilities to binary class predictions
        y_pred = [1 if p >= 0.5 else 0 for p in X_preds]
        # Generate classification report
        classification = classification_report(y_labels, y_pred, target_names=['Not Fraud', 'Fraud'], output_dict=True)
        # Dict to json
        classification_str = json.dumps(classification)
        return loss, len(self.valloader), {"ROC_AUC": ROC_AUC, "AUC": AUC, "Classification_str": classification_str, "Loss": loss}


def client_fn(context: Context):
    # Load model and data
    net = Net(input_dim, dim_model, num_classes, num_heads)
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)