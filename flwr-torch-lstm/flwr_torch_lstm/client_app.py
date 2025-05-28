"""flwr-torch-lstm: A Flower / PyTorch app."""

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr_torch_lstm.task import Net, get_weights, load_data, set_weights, test, train
import json
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, classification_report
import numpy as np


## Hyper-parameters 
input_size = 16 # dataset collumns
hidden_size = 1
num_layers = 3
num_classes = 2 # num y class

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train_loss = train(
            net=self.net,
            trainloader=self.trainloader,
            epochs=self.local_epochs,
            device=self.device,
        )
        return (
            get_weights(self.net),
            len(self.trainloader),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy, X_preds, y_labels= test(self.net, self.testloader, self.device)
        # Precision-Recall curve and ROC-AUC score
        precision, recall, thresholds = precision_recall_curve(y_labels, X_preds)
        ROC_AUC = roc_auc_score(y_labels, X_preds)
        AUC = auc(recall, precision)
        # Convert probabilities to binary class predictions
        y_pred = [1 if p >= 0.5 else 0 for p in X_preds]
        print ("HERE", X_preds[1])
        print ("HERE2", X_preds[2])
        print ("HERE3", X_preds[3])
        print ("HERE4", y_labels[1])
        print ("HERE5", y_labels[2])
        print ("HERE6", y_labels[3])
        # print ("precision: ",precision[0])
        # print ("recall: ",recall[0])
        # print ("y_pred: ",y_pred[0])
        # Generate classification report
        classification = classification_report(y_labels, y_pred, target_names=['Not Fraud', 'Fraud'], output_dict=True)
        # print("y_labels:", np.unique(y_labels, return_counts=True))
        # print("y_pred:", np.unique(y_pred, return_counts=True))
        # Dict to json
        classification_str = json.dumps(classification)
        return loss, len(self.testloader), {"ROC_AUC": ROC_AUC, "AUC": AUC, "Classification_str": classification_str}
        # return loss, len(self.testloader), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = Net(input_size, hidden_size, num_layers, num_classes)
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, testloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, testloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)