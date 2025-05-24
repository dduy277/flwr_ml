"""flwr-sklearn-LogisticRegression: A Flower / sklearn app."""

import json
import pandas as pd
from typing import List, Tuple
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr_sklearn_logisticregression.task import get_model, get_model_params, set_initial_params, set_model_params
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, log_loss, classification_report
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.data.pandas_dataset import PandasDataset

"""MlFlow tracking"""
# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://localhost:5000")

# # Create a new MLflow Experiment
mlflow.set_experiment("MLflow Quickstart")
mlflow.start_run()

# Take ROC_AUC, AUC, classification_report
def avg_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    ROC_AUC=[]; AUC=[]; precision=[]; recall=[]; f1_score=[]
    """A func that aggregates metrics"""

    for _,m in metrics:
        "Get ROC_AUC, AUC"
        # get metric
        ROC_AUC_temp = m.get("ROC_AUC")
        AUC_temp = m.get("AUC")
        # put metrics into array
        ROC_AUC.append(ROC_AUC_temp)
        AUC.append(AUC_temp)
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

    # np.float64 doesn't affect anything, it looks ugly though.
    # return {"precision": precision, "recall": recall, "f1-score": f1_score, "ROC_AUC": ROC_AUC, "AUC": AUC}
    return {"avg_precision": avg_precision, "avg_recall": avg_recall, "avg_f1_score": avg_f1_score, "avg_ROC_AUC": avg_ROC_AUC, "avg_AUC": avg_AUC}


# Evaluates the global mode
def get_eval_func(X_test_global, y_test_global, g_model, num_rounds, params, Test_ds):
    """Return a callback that evaluates the global model"""
    def eval(server_round, parameters_ndarrays, config): # server_round == current round
        set_model_params(g_model, parameters_ndarrays)
        # Eval
        loss = log_loss(y_test_global, g_model.predict_proba(X_test_global))
        precision, recall, thresholds = precision_recall_curve(y_test_global, y_score=g_model.predict_proba(X_test_global)[:, 1])
        ROC_AUC = roc_auc_score(y_test_global, g_model.predict_proba(X_test_global)[:, 1])
        AUC = auc(recall, precision)
        classification = classification_report(y_test_global, g_model.predict(X_test_global), target_names=['Not Fraud', 'Fraud'], zero_division=0, output_dict=True)

        ROC_AUC = round(ROC_AUC, 4)
        AUC = round(AUC, 4)
        precision = round(classification.get('Fraud', {}).get('precision'), 2)
        recall = round(classification.get('Fraud', {}).get('recall'), 2)
        f1_score = round(classification.get('Fraud', {}).get('f1-score'), 2)

        # Log the metrics (last run)
        if server_round == num_rounds:
            # Log metric, params
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1-score", f1_score)
            mlflow.log_metric("ROC_AUC", ROC_AUC)
            mlflow.log_metric("AUC", AUC)
            mlflow.log_metric("Loss", loss)
            mlflow.log_params(params)
            # Log test dataset
            mlflow.log_input(Test_ds, context="testing")
            # Log the model
            signature = infer_signature(X_test_global, g_model.predict(X_test_global))
            mlflow.sklearn.log_model(
            sk_model=g_model, 
            artifact_path="G_model", 
            signature=signature, 
            registered_model_name="Gobal_flwr-sklearn-logisticregression", 
            input_example=X_test_global,
            )
            mlflow.end_run()
            # # End MLflow Experiment
        return loss, {"precision": precision, "recall": recall, "f1-score": f1_score, "ROC_AUC": ROC_AUC, "AUC": AUC}
    
    # Log the model
    return eval


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Create LogisticRegression Model
    penalty = context.run_config["penalty"]
    local_epochs = context.run_config["local-epochs"]
    model = get_model(penalty, local_epochs)
    params = {
    "penalty":penalty,
    "max_iter":local_epochs,
    "warm_start":False,
    }
    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(model)
    initial_parameters = ndarrays_to_parameters(get_model_params(model))
    
    # # Load global test set
    df_test = pd.read_csv('../ML/CSV/df_test_3.csv')
    df_test.drop("Unnamed: 0", axis=1, inplace=True)
    # ".values" to fix: X has feature names, but LogisticRegression was fitted without feature names
    X_test_global = df_test.drop('Class', axis=1).values
    y_test_global = df_test['Class'].values
    Test_ds: PandasDataset = mlflow.data.from_pandas(df_test, targets="Class") # for MLflow

    # Define strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=avg_metrics,
        evaluate_fn=get_eval_func(X_test_global, y_test_global, model, num_rounds, params, Test_ds),
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
