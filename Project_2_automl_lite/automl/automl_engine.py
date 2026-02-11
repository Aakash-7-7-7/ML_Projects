from automl.models import (
    LinearRegressionModel,
    RandomForestModel,
    GradientBoostingModel
)
from automl.evaluator import Evaluator
from automl.experiment_tracker import ExperimentTracker


class AutoMLEngine:
    def __init__(self, experiment_name: str):
        self.models = [
            LinearRegressionModel(),
            RandomForestModel(),
            GradientBoostingModel()
        ]
        self.evaluator = Evaluator()
        self.tracker = ExperimentTracker(experiment_name)
        self.best_model = None
        self.best_score = float("inf")  # lower RMSE is better

    def run(self, X_train, X_test, y_train, y_test, preprocessor):
        """
        Runs AutoML process:
        - trains models
        - evaluates
        - logs experiments
        - selects best model
        """

        for model in self.models:
            model_name = model.get_name()

            print(f"\nTraining {model_name}...")

            self.tracker.start_run(run_name=model_name)

            # Create full pipeline (preprocessing + model)
            from sklearn.pipeline import Pipeline

            full_pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("model", model.model)
            ])

            # Train
            full_pipeline.fit(X_train, y_train)

            # Predict
            predictions = full_pipeline.predict(X_test)

            # Evaluate
            metrics = self.evaluator.evaluate(y_test, predictions)

            print(f"{model_name} Results:", metrics)

            # Log to MLflow
            self.tracker.log_params({"model_name": model_name})
            self.tracker.log_metrics(metrics)
            self.tracker.log_model(full_pipeline)

            self.tracker.end_run()

            # Track best model
            if metrics["rmse"] < self.best_score:
                self.best_score = metrics["rmse"]
                self.best_model = full_pipeline

        print("\nBest model selected with RMSE:", self.best_score)

        return self.best_model
