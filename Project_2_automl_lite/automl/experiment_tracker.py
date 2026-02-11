import mlflow
import mlflow.sklearn

class ExperimentTracker:
    def __init__(self,experiment_name:str):
        mlflow.set_experiment(experiment_name)

    def start_run(self,run_name:str):
        mlflow.start_run(run_name=run_name)

    def log_params(self,params:dict):
        for key , value in params.items():
            mlflow.log_param(key,value)

    def log_metrics(self,metrics:dict):
        for key, value in metrics.items():
            mlflow.log_metric(key,value)

    def log_model(self,model,artifact_path="model"):
        mlflow.sklearn.log_model(model,artifact_path)

    def end_run(self):
        mlflow.end_run()