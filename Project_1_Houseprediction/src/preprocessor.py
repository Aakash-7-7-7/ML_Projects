import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts

class Preprocessor:
    def __init__(self):
        self.scaler=StandardScaler()

    def split_data(self,data:pd.DataFrame):
        ''' 
        #!Spliting the data
        '''
        X=data[['Gr Liv Area','Year Built']]
        y=data['SalePrice']
        return X,y

    def train_test_split(self,X,y,test_size=0.2):
        '''
        #!Splitting the data
        '''
        return tts(X,y,test_size=test_size,random_state=42)
    
    def scale_data(self,X_train,X_test):
        ''' 
        #! Standardizing the data
        '''
        X_train_scaled=self.scaler.fit_transform(X_train)
        X_test_scaled=self.scaler.transform(X_test)

        return X_train_scaled,X_test_scaled
    
