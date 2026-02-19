from abc import ABC,abstractmethod
from sklearn.linear_model import LinearRegression
import numpy as np 


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


class LinearModel(BaseModel):

    def __init__(self):
        self.model=LinearRegression()

    def train(self, X_train, y_train):
        self.model.fit(X_train,y_train)

    def predict(self, X):
        return self.model.predict(X)
    
    def get_name(self):
        return "Linear Regression"

    
    
# ----------- Usage -----------
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

model=LinearModel()
model.train(X,y)

pred=model.predict([[5]])
print('Model predicted',pred)
print("Model_name",model.get_name())