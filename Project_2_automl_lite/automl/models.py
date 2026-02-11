from abc import ABC,abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC

class BaseModel(ABC):

    @abstractmethod
    def train(self,X_train,y_train):
        pass

    @abstractmethod
    def predict(self,X):
        pass

    @abstractmethod
    def get_name(self):
        pass

#-----------------------
# Linear Regression
#-----------------------

class LinearRegressionModel(BaseModel):
    def __init__(self):
        self.model=LinearRegression()

    def train(self,X_train,y_train):
        self.model.fit(X_train,y_train)

    def predict(self,X):
        return self.model.predict(X)
    
    def get_name(self):
        return "LinearRegression"
    

#-------------------------
#SVC
#-------------------------
class SupportVectorModel(ABC):

    def __init__(self):
        self.model=SVC(
            kernel='rbf',
            degree=3,
            decision_function_shape='ovr'
        )

    def train(self,X_train,y_train):
        self.model.fit(X_train,y_train)

    def predict(self,X):
        self.model.predict(X)

    def get_name(self):
        return "Support Vector Machine"
    

# ----------------------------
# Random Forest
# ----------------------------
class RandomForestModel(BaseModel):
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            random_state=42
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def get_name(self):
        return "RandomForest"


# ----------------------------
# Gradient Boosting
# ----------------------------
class GradientBoostingModel(BaseModel):
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def get_name(self):
        return "GradientBoosting"
