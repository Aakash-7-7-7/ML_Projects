from sklearn.metrics import mean_squared_error,r2_score

class Trainer:
    def __init__(self,model):
        self.model=model

    def train(self,X_train,y_train):
        ''' 
        #! Train the Ml model
        '''
        self.model.train(X_train,y_train)

    def evaluate(self,X_test,y_test):
        ''' 
        #! Evaluate the model
        '''
        prediction=self.model.predict(X_test)
        mse=mean_squared_error(y_test,prediction)
        r2=r2_score(y_test,prediction)

        return{
            "mse":mse,
            "r2_score":r2
        }