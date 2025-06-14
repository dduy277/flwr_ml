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
import math



def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, dim_model, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.head_dim = dim_model // num_heads
        
        self.qkv_layer = nn.Linear(input_dim , 3 * dim_model)
        self.linear_layer = nn.Linear(dim_model, dim_model)
    
    def forward(self, x, mask):
        batch_size, sequence_length, input_dim = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        out = self.linear_layer(values)
        return out, values


device = torch.device("xpu:0" if torch.xpu.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, input_dim, dim_model, num_classes, num_heads):
        super(Net, self).__init__()
        self.multihead_attention = MultiheadAttention(input_dim=input_dim, dim_model=dim_model, num_heads=num_heads)
        self.fc = nn.Linear(dim_model, num_classes)
    def forward(self, x, mask=None):
        # Apply Multihead Attention
        attn_output, _= self.multihead_attention(x, mask)  # [batch_size, seq_len, input_dim]
        out = attn_output[:, -1, :]  # [batch_size, (remove), input_dim]
        out = self.fc(out)
        return out

def load_data(partition_id: int, num_partitions: int):
    """Load partitioned dataset."""
    df = pd.read_csv('CSV/df_train_3.csv')
    df.drop("Unnamed: 0", axis=1, inplace=True)
    dataset = Dataset.from_pandas(df)
    partitioner = IidPartitioner(num_partitions=num_partitions)
    partitioner.dataset = dataset
    dataset = partitioner.load_partition(partition_id=partition_id).to_pandas()
    dataset = dataset.astype('float32')
    # Split the data: 80% train, 20% test
    trainloader, testloader = train_test_split(dataset, test_size=0.2, random_state=42, stratify=dataset['Class'])
    return trainloader, testloader

def train(net, trainloader, epochs, device):
    """Train the model on the training set"""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    running_loss = 0.0
    for epoch in range(epochs):
        # Extract features and labels once per epoch
        X_train = trainloader.drop('Class', axis=1).values
        X_train = torch.from_numpy(np.expand_dims(X_train, axis=2)).to(device)
        y_train = torch.from_numpy(trainloader['Class'].values).long().to(device)

        # Forward pass
        outputs = net(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_trainloss = running_loss / epochs
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
            X_test = torch.from_numpy(np.expand_dims(X_test, axis=2)).float()
            y_test = torch.from_numpy(testloader['Class'].values).long()
            
            outputs = net(X_test.to(device))
            y_test = y_test.to(device)
            loss += criterion(outputs, y_test).item()

            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Probability for the positive class
            all_X_preds.extend(probs)
            all_y_labels.extend(y_test.cpu().numpy())

            correct += (torch.max(outputs.data, 1)[1] == y_test).sum().item()
    accuracy = correct / len(testloader)
    loss = loss / len(testloader)
    return loss, accuracy, all_X_preds, all_y_labels


def test(net, testloader, device):
    """Validate the model on the test set"""
    net.to(device)  # move model to GPU if available
    net.eval()  # Set to evaluation mode
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0
    all_X_preds = []
    all_y_labels = []
    with torch.no_grad():
        # Extract features and labels once
        X_test = testloader.drop('Class', axis=1).values
        X_test = torch.from_numpy(np.expand_dims(X_test, axis=2)).to(device)
        y_test = torch.from_numpy(testloader['Class'].values).long().to(device)
        outputs = net(X_test)
        loss = criterion(outputs, y_test).item()
        # Get probabilities for the positive class (class 1)
        probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        all_X_preds.extend(probs)
        all_y_labels.extend(y_test.cpu().numpy())
        # accuracy
        correct = (torch.max(outputs.data, 1)[1] == y_test).sum().item()
    loss = loss / len(testloader)
    accuracy = correct / len(testloader)
    return loss, accuracy, np.array(all_X_preds), np.array(all_y_labels)


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
