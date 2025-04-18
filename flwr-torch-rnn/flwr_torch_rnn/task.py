"""flwr-torch-rnn: A Flower / PyTorch app."""

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets.partitioner import IidPartitioner
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, log_loss, classification_report

 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Net, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, h_n = self.rnn(x,h0)
        # out = batch_size, seq_legnth, hidden_size
        out = out [:, -1, :]
        # out = batch_size, hidden_size
        out = self.fc(out)
        return out
 

fds = None  # Cache FederatedDataset

def load_data(partition_id: int, num_partitions: int):
    """Load partition df_3 data."""
    df = pd.read_csv('/home/zuy/Documents/BCU/ML/CSV/df_train_3.csv')
    df.drop("Unnamed: 0", axis=1, inplace=True)
    dataset = Dataset.from_pandas(df)
    partitioner = IidPartitioner(num_partitions=num_partitions)
    # partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by="Class", alpha=10, min_partition_size=5)
    partitioner.dataset = dataset
    dataset = partitioner.load_partition(partition_id=partition_id).to_pandas()
    dataset = dataset.astype('float32')
    # Split the on edge data: 80% train, 20% test
    trainloader, testloader= train_test_split(dataset, test_size=0.2, random_state=42, stratify=dataset['Class'])

    return trainloader, testloader


def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for i in trainloader:
            X_train = trainloader.drop('Class', axis=1).values
            X_train = torch.from_numpy(np.expand_dims(X_train, axis =1))  
            y_train = torch.from_numpy(trainloader['Class'].values).long()

            outputs = net(X_train.to(device))
            y_train = y_train.to(device)
            loss = criterion(outputs, y_train)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)# move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0
    all_X_preds = []
    all_y_labels = []
    with torch.no_grad():
        net.eval()
        for i in testloader:
            X_test = testloader.drop('Class', axis=1).values
            X_test = torch.from_numpy(np.expand_dims(X_test, axis =1)).to(device)
            y_test = torch.from_numpy(testloader['Class'].values).long().to(device)
            # Forward pass
            outputs = net(X_test)
            loss += criterion(outputs, y_test).item()
            # Predictions
            _, predicted = torch.max(outputs, 1) 
            all_X_preds.extend(predicted.cpu().numpy())
            all_y_labels.extend(y_test.cpu().numpy())
            # Accuracy calculation
            correct += (predicted == y_test).sum().item()

    report = classification_report(all_X_preds, all_y_labels, target_names=['Not Fraud', 'Fraud'], output_dict=True, zero_division=0.0)
    precision = report['Fraud']['precision']
    recall = report['Fraud']['recall']
    f1 = report['Fraud']['f1-score']
    accuracy = correct / len(testloader)
    loss = loss / len(testloader)
    return loss, accuracy, precision, recall, f1


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
