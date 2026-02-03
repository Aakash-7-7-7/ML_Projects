from sklearn.linear_model import LinearRegression

class Housepred:
    def __init__(self):
        self.model=LinearRegression()

    def train(self,X_train,y_train):
        '''
        #! Train Linear Model
        '''
        self.model.fit(X_train,y_train)

    def predict(self,X):

        '''
        #! Predict Price
        '''
        return self.model.predict(X)