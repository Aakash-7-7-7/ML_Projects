import numpy as np 
from sklearn.metrics import mean_squared_error,r2_score

class Evaluator:
    def evaluate(self,y_true,y_pred):

        mse=mean_squared_error(y_true,y_pred)
        rmse=np.sqrt(mse)
        r2=r2_score(y_true,y_pred)

        result={
            "rmse":rmse,
            "r2_score":r2
        }
        return result