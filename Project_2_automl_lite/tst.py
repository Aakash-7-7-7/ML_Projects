import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class Preprocessor:
    def __init__(self, target_column: str):
        self.target_column = target_column

    def split_features_target(self, df: pd.DataFrame):
        """
        Splits dataframe into X (features) and y (target)
        """
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        return X, y

    def train_test_split(self, X, y, test_size=0.2, random_state=42):
        """
        Splits data into train and test sets
        """
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    def build_pipeline(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Builds preprocessing pipeline for numerical and categorical features
        """
        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
        categorical_features = X.select_dtypes(include=["object"]).columns

        numeric_pipeline = Pipeline(steps=[
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline(steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features)
            ]
        )

        return preprocessor
