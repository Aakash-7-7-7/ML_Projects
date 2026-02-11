from automl.data_loader import DataLoader
from automl.preprocessor import Preprocessor
from automl.automl_engine import AutoMLEngine


# Load data
loader = DataLoader("data/housing.csv")
df = loader.load_data()

# Preprocess
preprocessor_obj = Preprocessor(target_column="median_house_value")
X, y = preprocessor_obj.split_features_target(df)
X_train, X_test, y_train, y_test = preprocessor_obj.train_test_split(X, y)
preprocessor = preprocessor_obj.build_pipeline(X_train)

# Run AutoML
engine = AutoMLEngine(experiment_name="Housing_AutoML_Trial")

best_model = engine.run(
    X_train, X_test, y_train, y_test, preprocessor
)

print("\nAutoML process completed.")
