# PyTorch RNN Model with MLflow Integration
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, classification_report
from sklearn.model_selection import train_test_split
import time
import sys
sys.path.insert(0, "flwr-torch-rnn")
sys.path.insert(1, "flwr-torch-gru")
sys.path.insert(2, "flwr-torch-lstm")
sys.path.insert(3, "flwr-torch-multiheadattention")


mlflow.set_tracking_uri("http://localhost:5000")

def load_model_from_mlflow(model_name=None, model_version=None, run_id=None):
    
    # Load from model registry
    model_uri = f"models:/{model_name}/{model_version}"
    print(f"Loading model from registry: {model_uri}")
    loaded_model = mlflow.pytorch.load_model(model_uri)
    return loaded_model


def test_model(net, X_test, y_test, device, model_name):
    if "MultiheadAttention" in model_name:
        """Testing muti-head model"""
        print("Testing Muti-head Attention model")
        net.to(device)
        net.eval()
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            X_test_tensor = torch.from_numpy(np.expand_dims(X_test, axis=1)).to(device)
            y_test_tensor = torch.from_numpy(y_test).long().to(device)
            
            outputs = net(X_test_tensor)
            loss = criterion(outputs, y_test_tensor).item()
            
            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            predictions = torch.max(outputs.data, 1)[1]
            correct = (predictions == y_test_tensor).sum().item()
            accuracy = correct / len(y_test)
        return loss, accuracy, probs, predictions.cpu().numpy()
        
    else:
        """Testing rnn, lstm, gru model"""
        print("Testing rnn, lstm, gru model")
        net.to(device)
        net.eval()
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            X_test_tensor = torch.from_numpy(np.expand_dims(X_test, axis=1)).to(device)
            y_test_tensor = torch.from_numpy(y_test).long().to(device)
            outputs = net(X_test_tensor)
            loss = criterion(outputs, y_test_tensor).item()
            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            predictions = torch.max(outputs.data, 1)[1]
            correct = (predictions == y_test_tensor).sum().item()
            accuracy = correct / len(y_test)
        return loss, accuracy, probs, predictions.cpu().numpy()


device = torch.device('cpu')

# Load model from MLflow
model_name=input('Input model name: ')

# Start MLflow run
mlflow.set_experiment("Deploy rnn df2")
mlflow.start_run(run_name=f"Deploy_scenal_2_fhe-rnn")

model = load_model_from_mlflow(model_name=model_name, model_version="latest")
if model is None:
    raise RuntimeError("Failed to load model from MLflow. Please check your MLflow server and model availability.")

# Load df
if "1" in model_name:
    # Load dataset
    print("Load df_1")
    df = pd.read_csv('CSV/df_1.csv')
    df.drop("Unnamed: 0", axis=1, inplace=True)
    target_col = 'Class'
elif "2" in model_name:
    # Load dataset
    print("Load df_2")
    df = pd.read_csv('CSV/df_2.csv')
    df.drop("Unnamed: 0", axis=1, inplace=True)
    target_col = 'isFraud'

# Split data: Full, 50%, 20%, 10%
df_50_persent, df_temp = train_test_split(df, test_size=0.5, random_state=42, stratify=df[target_col])
df_20_persent, df_temp_2 = train_test_split(df_temp, test_size=0.6, random_state=42, stratify=df_temp[target_col])  # 0.6 of 50% = 20% of original
df_10_persent, _ = train_test_split(df_temp_2, test_size=0.5, random_state=42, stratify=df_temp_2[target_col])  # 0.5 of 20% = 10% of original

# Test datasets with different sizes
test_sets = [
    ('full', df),
    ('50', df_50_persent), 
    ('20', df_20_persent),
    ('10', df_10_persent)
]

for size_name, test_df in test_sets:
    # Prepare test data
    X_test = test_df.drop(target_col, axis=1).astype('float32').values
    y_test = test_df[target_col].astype('float32').values
    
    # Test the loaded model
    start_ts = time.time()
    loss, accuracy, X_preds, y_labels = test_model(model, X_test, y_test, device, model_name)
    end_ts = time.time()
    time_taken = round(end_ts - start_ts, 3)
    
    y_pred = [1 if p >= 0.5 else 0 for p in X_preds]
    classification = classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud'], output_dict=True)
    precision = round(classification.get('Fraud', {}).get('precision', 0), 2)
    recall = round(classification.get('Fraud', {}).get('recall', 0), 2)
    f1_score = round(classification.get('Fraud', {}).get('f1-score', 0), 2)

    print(f"precision_{size_name}:", precision)
    print(f"recall_{size_name}:", recall)
    print(f"f1_score_{size_name}:", f1_score)
    
    # Log metrics
    mlflow.log_metric(f"time_{size_name}", time_taken)
    mlflow.log_metric(f"precision_{size_name}", precision)
    mlflow.log_metric(f"recall_{size_name}", recall)
    mlflow.log_metric(f"f1_score_{size_name}", f1_score)

mlflow.end_run()