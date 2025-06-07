"""flwr-torch-MultiheadAttention: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr_torch_multiheadattention.task import Net, get_weights, set_weights, test
import json
import pandas as pd
from typing import List, Tuple
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, log_loss, classification_report
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.data.pandas_dataset import PandasDataset
import torch
import logging



# """MlFlow tracking"""
# # Set our tracking server uri for logging
# mlflow.set_tracking_uri(uri="http://localhost:5000")

# # Set log level to debugging
# # # (MLflow can't verifile input data, so turm off the debug for now)
# logger = logging.getLogger("mlflow")
# logger.setLevel(logging.NOTSET)

# # Create / start a new MLflow Experiment
# mlflow.set_experiment("MLflow Quickstart")
# mlflow.start_run(run_name = "Gobal_flwr-torch-MultiheadAttention")


## Hyper-parameters 
input_dim = 1
dim_model = 64
num_classes = 2 # num y class
num_heads = 4


# Get device (need to be global ?)
device = torch.device("xpu:0" if torch.xpu.is_available() else "cpu")


# Take ROC_AUC, AUC, classification_report
def avg_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    ROC_AUC=[]; AUC=[]; precision=[]; recall=[]; f1_score=[]; loss=[]
    """A func that aggregates metrics"""

    for _,m in metrics:
        "Get ROC_AUC, AUC"
        # get metric
        ROC_AUC_temp = m.get("ROC_AUC")
        AUC_temp = m.get("AUC")
        loss_temp=m.get("Loss")
        # put metrics into array
        ROC_AUC.append( round(ROC_AUC_temp, 4) )
        AUC.append( round(AUC_temp, 4) )
        loss.append( round(loss_temp, 4) )
        # average of metrics
        avg_ROC_AUC = round(sum(ROC_AUC) / len(ROC_AUC), 4)
        avg_AUC = round(sum(AUC) / len(AUC), 4)
        "Get classification_str"
        # json to dict
        classification = json.loads(m["Classification_str"])
        # get metric
        precision_temp = classification.get('Fraud', {}).get('precision')
        recall_temp = classification.get('Fraud', {}).get('recall')
        f1_score_temp = classification.get('Fraud', {}).get('f1-score')
        # put metrics into array
        precision.append(round(precision_temp, 2))
        recall.append(round(recall_temp, 2))
        f1_score.append(round(f1_score_temp, 2))

    # average of metrics
    avg_precision = round(sum(precision) / len(precision), 2)
    avg_recall = round(sum(recall) / len(recall), 2)
    avg_f1_score = round(sum(f1_score) / len(f1_score), 2)

    return {"precision": precision, "recall": recall, "f1-score": f1_score, "ROC_AUC": ROC_AUC, "AUC": AUC, "loss": loss}
    # return {"avg_precision": avg_precision, "avg_recall": avg_recall, "avg_f1_score": avg_f1_score, "avg_ROC_AUC": avg_ROC_AUC, "avg_AUC": avg_AUC}


# Evaluates the global mode
def get_eval_func(valloader, g_model, num_rounds, params, Test_ds):
    """Return a callback that evaluates the global model"""
    def eval(server_round, parameters_ndarrays, config): # server_round == current round
        set_weights(g_model, parameters_ndarrays)
        X_test_global = valloader.drop('Class', axis=1).values
        y_test_global = valloader['Class'].values
        input_example = valloader.drop('Class', axis=1)
        # Eval
        loss, accuracy, X_preds, y_labels= test(g_model, valloader, device)
        # Precision-Recall curve and ROC-AUC score
        precision, recall, thresholds = precision_recall_curve(y_labels, X_preds)
        ROC_AUC = roc_auc_score(y_labels, X_preds)
        AUC = auc(recall, precision)
        # Convert probabilities to binary class predictions
        y_pred = [1 if p >= 0.5 else 0 for p in X_preds]
        # Generate classification report
        classification = classification_report(y_labels, y_pred, target_names=['Not Fraud', 'Fraud'], output_dict=True)

        ROC_AUC = round(ROC_AUC, 4)
        AUC = round(AUC, 4)
        precision = round(classification.get('Fraud', {}).get('precision'), 2)
        recall = round(classification.get('Fraud', {}).get('recall'), 2)
        f1_score = round(classification.get('Fraud', {}).get('f1-score'), 2)

        # # Log the metrics (final run only)
        # if server_round == num_rounds:
        #     # Log metric, params
        #     mlflow.log_metric("precision", precision)
        #     mlflow.log_metric("recall", recall)
        #     mlflow.log_metric("f1-score", f1_score)
        #     mlflow.log_metric("ROC_AUC", ROC_AUC)
        #     mlflow.log_metric("AUC", AUC)
        #     mlflow.log_metric("Loss", loss)
        #     mlflow.log_params(params)
        #     # Log test dataset
        #     mlflow.log_input(Test_ds, context="testing")
        #     # Log the model
        #     signature = infer_signature(X_test_global, y_pred)
        #     mlflow.pytorch.log_model(
        #     pytorch_model=g_model, 
        #     artifact_path="G_model", 
        #     signature=signature,
        #     registered_model_name="Gobal_flwr-torch-MultiheadAttention", 
        #     input_example= input_example.iloc[[0]],
        #     )
        #     mlflow.end_run()    # End MLflow logging
        return loss, {"precision": precision, "recall": recall, "f1-score": f1_score, "ROC_AUC": ROC_AUC, "AUC": AUC}
    
    return eval


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    local_epochs = context.run_config["local-epochs"]
    # Initialize model parameters
    ndarrays = get_weights(Net(input_dim, dim_model, num_classes, num_heads))
    parameters = ndarrays_to_parameters(ndarrays)
    params = {
    "local_epochs":local_epochs,
    "input_dim": input_dim,
    "dim_model": dim_model,
    "num_classes": num_classes, # num y class
    "num_heads": num_heads,
    }
    
    # Load model
    g_model = Net(input_dim, dim_model, num_classes, num_heads)

    # # Load global test set
    valloader = pd.read_csv('CSV/df_test_3.csv')
    valloader.drop("Unnamed: 0", axis=1, inplace=True)
    valloader = valloader.astype('float32')
    
    # ".values" to fix: X has feature names, but LogisticRegression was fitted without feature names
    # Split the on edge data: 80% train, 20% test
    Test_ds: PandasDataset = mlflow.data.from_pandas(valloader, targets="Class") # for MLflow


    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=avg_metrics,
        evaluate_fn=get_eval_func(valloader, g_model, num_rounds, params, Test_ds),
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
