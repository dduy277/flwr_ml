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
mlflow.set_experiment("Deploy rnn df1")
mlflow.start_run(run_name = "Deploy_scenal_2_fhe-rnn", log_system_metrics=True)

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
            X_test_tensor = torch.from_numpy(np.expand_dims(X_test, axis=2)).to(device)
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
model = load_model_from_mlflow(model_name=model_name, model_version="latest")
if model is None:
    raise RuntimeError("Failed to load model from MLflow. Please check your MLflow server and model availability.")

# Load df
if "1" in model_name:
    # Load dataset
    print("Load df_1")
    df = pd.read_csv('CSV/df_1.csv')
    df.drop("Unnamed: 0", axis=1, inplace=True)

    df_50_persent, df_temp = train_test_split(df, test_size=0.5, random_state=42, stratify=df.Class)
    df_10_persent, df_temp_2 = train_test_split(df_temp, test_size=0.2, random_state=42, stratify=df_temp.Class)
    df_1_val = df_temp_2[df_temp_2['Class'] == 1].iloc[0]

    # Full
    X_test = df.drop('Class', axis=1)
    y_test = df['Class']
    # Convert data for testing
    X_test = X_test.astype('float32').values
    y_test = y_test.astype('float32').values

    # 50
    X_test_50 = df_50_persent.drop('Class', axis=1)
    y_test_50 = df_50_persent['Class']
    # Convert data for testing
    X_test_50 = X_test_50.astype('float32').values
    y_test_50 = y_test_50.astype('float32').values

    # 10
    X_test_10 = df_10_persent.drop('Class', axis=1)
    y_test_10 = df_10_persent['Class']
    # Convert data for testing
    X_test_10 = X_test_10.astype('float32').values
    y_test_10 = y_test_10.astype('float32').values

    # 1
    X_test_1 = df_1_val.drop('Class', axis=0)
    y_test_1 = df_1_val['Class']
    # Convert data for testing
    X_test_1 = X_test_1.astype('float32').values
    y_test_1 = y_test_1.astype('float32')
elif "2" in model_name:
    # Load dataset
    print("Load df_2")
    df = pd.read_csv('CSV/df_2.csv')
    df.drop("Unnamed: 0", axis=1, inplace=True)

    df_50_persent, df_temp = train_test_split(df, test_size=0.5, random_state=42, stratify=df.Class)
    df_10_persent, df_temp_2 = train_test_split(df_temp, test_size=0.2, random_state=42, stratify=df_temp.Class)
    df_1_val = df_temp_2[df_temp_2['isFraud'] == 1].iloc[0]

    # Full
    X_test = df.drop('isFraud', axis=1)
    y_test = df['isFraud']
    # Convert data for testing
    X_test = X_test.astype('float32').values
    y_test = y_test.astype('float32').values

    # 50
    X_test_50 = df_50_persent.drop('isFraud', axis=1)
    y_test_50 = df_50_persent['isFraud']
    # Convert data for testing
    X_test_50 = X_test_50.astype('float32').values
    y_test_50 = y_test_50.astype('float32').values

    # 10
    X_test_10 = df_1_val.drop('isFraud', axis=1)
    y_test_10 = df_1_val['isFraud']
    # Convert data for testing
    X_test_10 = X_test_10.astype('float32').values
    y_test_10 = y_test_10.astype('float32').values

    X_test_1 = df_10_persent.drop('isFraud', axis=0)
    y_test_1 = df_10_persent['isFraud']
    # Convert data for testing
    X_test_1 = X_test_1.astype('float32').values
    y_test_1 = y_test_1.astype('float32')

# Test the loaded model
start_ts = time.time()
loss, accuracy, X_preds, y_labels = test_model(model, X_test, y_test, device, model_name)
end_ts = time.time()
time_full = round(end_ts - start_ts, 3)

y_pred = [1 if p >= 0.5 else 0 for p in X_preds]
classification = classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud'], output_dict=True)
precision = round(classification.get('Fraud', {}).get('precision'), 2)
recall = round(classification.get('Fraud', {}).get('recall'), 2)
f1_score = round(classification.get('Fraud', {}).get('f1-score'), 2)

print("precision_full:", precision)
print("recall_full:", recall)
print("f1_score_full:", f1_score)

# Test the loaded model 50_persent
start_ts = time.time()
loss, accuracy, X_preds, y_labels = test_model(model, X_test_50, y_test_50, device, model_name)
end_ts = time.time()
time_50 = round(end_ts - start_ts, 3)
y_pred = [1 if p >= 0.5 else 0 for p in X_preds]
classification = classification_report(y_test_50, y_pred, target_names=['Not Fraud', 'Fraud'], output_dict=True)
precision = round(classification.get('Fraud', {}).get('precision'), 2)
recall = round(classification.get('Fraud', {}).get('recall'), 2)
f1_score = round(classification.get('Fraud', {}).get('f1-score'), 2)

print("precision_50:", precision)
print("recall_50:", recall)
print("f1_score_50:", f1_score)

# Test the loaded model 10_persent
start_ts = time.time()
loss, accuracy, X_preds, y_labels = test_model(model, X_test, y_test, device, model_name)
end_ts = time.time()
time_10 = round(end_ts - start_ts, 3)

y_pred = [1 if p >= 0.5 else 0 for p in X_preds]
classification = classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud'], output_dict=True)
precision = round(classification.get('Fraud', {}).get('precision'), 2)
recall = round(classification.get('Fraud', {}).get('recall'), 2)
f1_score = round(classification.get('Fraud', {}).get('f1-score'), 2)

print("precision_10:", precision)
print("recall_10:", recall)
print("f1_score_10:", f1_score)

# Test the loaded model 1_val
start_ts = time.time()
loss, accuracy, X_preds_1_val, y_labels = test_model(model, X_test, y_test, device, model_name)
end_ts = time.time()
time_1 = round(end_ts - start_ts, 3)

y_pred = [1 if p >= 0.5 else 0 for p in X_preds]
classification = classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud'], output_dict=True)
precision = round(classification.get('Fraud', {}).get('precision'), 2)
recall = round(classification.get('Fraud', {}).get('recall'), 2)
f1_score = round(classification.get('Fraud', {}).get('f1-score'), 2)

print("precision_1:", precision)
print("recall_1:", recall)
print("f1_score_1:", f1_score)

# Log metric, params
mlflow.log_metric("time_full", time_full)
mlflow.log_metric("time_50", time_50)
mlflow.log_metric("time_10", time_10)
mlflow.log_metric("time_1", time_1)
mlflow.log_metric("1_val_class", df_1_val['Class'])
mlflow.log_metric("1_val-predicted-class ", y_pred)