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
import math


class MultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):   #self, 128, 3)
        super(MultiheadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0   # check if embed_dim % by num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Define linear layers for query, key, value (Q,K,V)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):   # x = ( batch_size, seq_len, embed_dim )
        print("shape: ",x.shape)
        batch_size, seq_len, embed_dim = x.size()
        # Linear projections
        Q = self.q_linear(x)    # ( batch, seq_len, embed_dim )
        K = self.k_linear(x)    # ( batch, seq_len, embed_dim )
        V = self.v_linear(x)    # ( batch, seq_len, embed_dim )
        # Split into heads
        # [ After reshape: (batch_size, seq_len, num_heads, head_dim) ] 
        # [ After transpose: (batch_size, num_heads, seq_len, head_dim) ]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  #(B, H, S, D)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, S, S)
        attn = F.softmax(scores, dim=-1)  # (B, H, S, S)
        context = torch.matmul(attn, V)   # (B, H, S, D)

        # Merge all attention heads back into a single tensor.
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)  # (B, S, E)

        # Final linear layer
        out = self.fc(context)  # ( batch, seq_len, embed_dim )
        return out


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_heads=3):
        super(Net, self).__init__()
        self.multihead_attention = MultiheadSelfAttention(embed_dim=input_size, num_heads=num_heads)
        self.fc = nn.Linear(input_size, num_classes)
    def forward(self, x):
        # Apply Multihead Attention
        attn_output, _ = self.multihead_attention(x)  # [batch_size, seq_len, input_size]
        out = attn_output[:, -1, :]  # [batch_size, (remove), input_size]
        out = self.fc(out)
        return out

def load_data(partition_id: int, num_partitions: int):
    """Load partitioned dataset."""
    df = pd.read_csv('/home/zuy/Documents/BCU/ML/CSV/df_train_3.csv')
    df.drop("Unnamed: 0", axis=1, inplace=True)
    dataset = Dataset.from_pandas(df)
    partitioner = IidPartitioner(num_partitions=num_partitions)
    partitioner.dataset = dataset
    dataset = partitioner.load_partition(partition_id=partition_id).to_pandas()
    dataset = dataset.astype('float32')
    
    trainloader, testloader = train_test_split(dataset, test_size=0.2, random_state=42, stratify=dataset['Class'])
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
            X_train = torch.from_numpy(np.expand_dims(X_train, axis=1))  # Add sequence dimension
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
