"""flwr-torch-MultiheadAttention: A Flower / PyTorch app."""

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
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int = 4,
        num_layers: int = 1, 
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super(Net, self).__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # project raw features into the attention embedding space
        self.input_proj = nn.Linear(input_size, hidden_size)
        # build a stack of MultiheadAttention + optional feed‑forward
        self.attn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(
                nn.ModuleDict({
                    "mha": nn.MultiheadAttention(
                        embed_dim=hidden_size,
                        num_heads=num_heads,
                        dropout=dropout,
                        batch_first=True,
                        # inputs/outputs are (B, S, E)
                    ),
                    "ff": nn.Sequential(
                        nn.Linear(hidden_size, hidden_size * 4),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_size * 4, hidden_size),
                        nn.Dropout(dropout),
                    ),
                    "norm1": nn.LayerNorm(hidden_size),
                    "norm2": nn.LayerNorm(hidden_size),
                })
            )
        # final classifier
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        """x: (batch, seq_len, input_size)"""
        # project into embed space, (B, S, E)
        x = self.input_proj(x)                    
        # pass through attention layers
        for layer in self.attn_layers:
            # self‑attention, (B, S, E)
            attn_out, _ = layer["mha"](x, x, x)
            x = layer["norm1"](x + attn_out)      # residual + norm
            # feed‑forward, (B, S, E)
            ff_out = layer["ff"](x) 
            x = layer["norm2"](x + ff_out)        # residual + norm
        # pool across sequence (could also take x[:, -1] if need "last token")
        # (B, E)
        x = x.mean(dim=1)                         
        # classification head, (B, num_classes)
        logits = self.fc(x)
        return logits


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
        for i in testloader:
            X_test = testloader.drop('Class', axis=1).values
            X_test = torch.from_numpy(np.expand_dims(X_test, axis =1))
            y_test = torch.from_numpy(testloader['Class'].values).long()
            
            outputs = net(X_test.to(device))
            y_test = y_test.to(device)
            loss += criterion(outputs, y_test).item()

            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Probability for the positive class
            # _, predicted = torch.max(outputs.data, 1)
            all_X_preds.extend(probs)
            all_y_labels.extend(y_test.cpu().numpy())

            correct += (torch.max(outputs.data, 1)[1] == y_test).sum().item()
    accuracy = correct / len(testloader)
    loss = loss / len(testloader)
    return loss, accuracy, all_X_preds, all_y_labels


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
