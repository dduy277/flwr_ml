"""flwr-torch-MultiheadAttention: A Flower / PyTorch app."""

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr_torch_multiheadattention.task import Net, get_weights, load_data, set_weights, test, train
import json
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, classification_report



## Hyper-parameters 
input_dim = 1 # dataset collumns
dim_model = 64
num_classes = 2 # num y class
num_heads = 4

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        return (
            get_weights(self.net),
            len(self.trainloader),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy, X_preds, y_labels= test(self.net, self.valloader, self.device)
        # Precision-Recall curve and ROC-AUC score
        precision, recall, thresholds = precision_recall_curve(y_labels, X_preds)
        ROC_AUC = roc_auc_score(y_labels, X_preds)
        AUC = auc(recall, precision)
        # Convert probabilities to binary class predictions
        y_pred = torch.tensor([1 if p >= 0.5 else 0 for p in X_preds], dtype=torch.int64)
        # print ("precision: ",precision[0])
        # print ("recall: ",recall[0])
        # print ("y_labels: ",y_labels[0])
        # print ("y_pred: ",y_pred[0])
        # Generate classification report
        classification = classification_report(y_labels, y_pred, target_names=['Not Fraud', 'Fraud'], output_dict=True)
        # Dict to json
        classification_str = json.dumps(classification)
        return loss, len(self.valloader), {"ROC_AUC": ROC_AUC, "AUC": AUC, "Classification_str": classification_str}
        # return loss, len(self.valloader), {"accuracy": accuracy}


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