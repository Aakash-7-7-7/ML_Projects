import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class Trainer:
    def __init__(self, model,scaler):
        self.model = model
        self.scaler=scaler

    def train_and_log(self, X_train, y_train, X_test, y_test):
        """
        Train model and log everything to MLflow
        """
        with mlflow.start_run():
            joblib.dump(self.scaler,'scaler.pkl')
            mlflow.log_artifact('scaler.pkl')
            

            # Train
            self.model.train(X_train, y_train)

            # Predict
            predictions = self.model.predict(X_test)

            # Metrics
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            # Log metrics
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2_score", r2)

            # Log model
            mlflow.sklearn.log_model(
                self.model.model,
                artifact_path="house_price_model",
                registered_model_name="HousePriceModel"
            )

            return {
                "mse": mse,
                "r2_score": r2
            }
