# PyTorch RNN Model with MLflow Integration
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, classification_report
import sys
sys.path.insert(0, "flwr-torch-rnn")
sys.path.insert(1, "flwr-torch-gru")
sys.path.insert(2, "flwr-torch-lstm")
sys.path.insert(3, "flwr-torch-multiheadattention")


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLflow MultiheadAttention df1")
mlflow.start_run(run_name = "Deploy_flwr-torch-rnn", log_system_metrics=True)

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
model = load_model_from_mlflow(model_name=model_name, model_version="latest")
if model is None:
    raise RuntimeError("Failed to load model from MLflow. Please check your MLflow server and model availability.")

# Load dataset
df_test = pd.read_csv('CSV/df_1.csv')
df_test.drop("Unnamed: 0", axis=1, inplace=True)

X_test = df_test.drop('Class', axis=1)
y_test = df_test['Class']

# Convert data for testing
X_test = X_test.astype('float32').values
y_test = y_test.astype('float32').values

# Test the loaded model
loss, accuracy, X_preds, y_labels = test_model(model, X_test, y_test, device, model_name)
# Convert probabilities to binary class predictions
y_pred = [1 if p >= 0.5 else 0 for p in X_preds]
precision, recall, thresholds = precision_recall_curve(y_test, X_preds)
ROC_AUC = roc_auc_score(y_test, X_preds)
AUC = auc(recall, precision)
classification = classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud'], output_dict=True)
ROC_AUC = round(ROC_AUC, 4)
AUC = round(AUC, 4)
precision = round(classification.get('Fraud', {}).get('precision'), 2)
recall = round(classification.get('Fraud', {}).get('recall'), 2)
f1_score = round(classification.get('Fraud', {}).get('f1-score'), 2)

# Log metric, params
mlflow.log_metric("precision", precision)
mlflow.log_metric("recall", recall)
mlflow.log_metric("f1-score", f1_score)
mlflow.log_metric("ROC_AUC", ROC_AUC)
mlflow.log_metric("AUC", AUC)
mlflow.log_metric("Loss", loss)


print(classification)
print("ROC_AUC:", ROC_AUC)
print("AUC:", AUC)