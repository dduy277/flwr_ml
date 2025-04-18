"""flwr-torch-MultiheadAttention: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr_torch_multiheadattention.task import Net, get_weights
import json
import pandas as pd
from typing import List, Tuple
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, log_loss, classification_report



## Hyper-parameters 
input_size = 16 # dataset collumns
hidden_size = 1
num_layers = 3
num_classes = 2 # num y class


# # Take ROC_AUC, AUC, classification_report
# def avg_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
#     ROC_AUC=[]; AUC=[]; precision=[]; recall=[]; f1_score=[]
#     """A func that aggregates metrics"""

#     for _,m in metrics:
#         "Get ROC_AUC, AUC"
#         # get metric
#         ROC_AUC_temp = m.get("ROC_AUC")
#         AUC_temp = m.get("AUC")
#         # put metrics into array
#         ROC_AUC.append(ROC_AUC_temp)
#         AUC.append(AUC_temp)
#         # average of metrics
#         avg_ROC_AUC = round(sum(ROC_AUC) / len(ROC_AUC), 4)
#         avg_AUC = round(sum(AUC) / len(AUC), 4)

#         "Get classification_str"
#         # json to dict
#         classification = json.loads(m["Classification_str"])
#         # get metric
#         precision_temp = classification.get('Fraud', {}).get('precision')
#         recall_temp = classification.get('Fraud', {}).get('recall')
#         f1_score_temp = classification.get('Fraud', {}).get('f1-score')
#         # put metrics into array
#         precision.append(round(precision_temp, 2))
#         recall.append(round(recall_temp, 2))
#         f1_score.append(round(f1_score_temp, 2))
#         # average of metrics
#     avg_precision = round(sum(precision) / len(precision), 2)
#     avg_recall = round(sum(recall) / len(recall), 2)
#     avg_f1_score = round(sum(f1_score) / len(f1_score), 2)

#     # return {"precision": precision, "recall": recall, "f1-score": f1_score, "ROC_AUC": ROC_AUC, "AUC": AUC}
#     return {"avg_precision": avg_precision, "avg_recall": avg_recall, "avg_f1_score": avg_f1_score, "avg_ROC_AUC": avg_ROC_AUC, "avg_AUC": avg_AUC}



def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net(input_size, hidden_size, num_layers, num_classes))
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        # evaluate_metrics_aggregation_fn=avg_metrics,

    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
