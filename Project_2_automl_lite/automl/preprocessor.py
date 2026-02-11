import pandas as pd 
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


class Preprocessor:

    def __init__(self,target_column:str):
        self.target_column=target_column

    def split_features_target(self,df:pd.DataFrame):
        '''
        Split the data into X features and y features
        '''
        X=df.drop(columns=[self.target_column])
        y=df[self.target_column]

        return X,y
    
    def train_test_split(self,X,y,test_size=0.2):

        return tts(X,y,test_size=test_size,random_state=42)
    
    def build_pipeline(self,X:pd.DataFrame)->ColumnTransformer:

        numer_feat=X.select_dtypes(include=['float32','float64']).columns
        cat_feature=X.select_dtypes(include=['object']).columns

        numer_pipeline=Pipeline(steps=[("scaler",StandardScaler()),
                                       ('imputer',SimpleImputer(strategy='most_frequent'))])
        
        cat_pipeline=Pipeline(steps=[("encoder",OneHotEncoder(handle_unknown='ignore')),
                                     ('imputer',SimpleImputer(strategy='most_frequent'))])

        preprocessor=ColumnTransformer(
            transformers=[
                ('num',numer_pipeline,numer_feat),
                ('cat',cat_pipeline,cat_feature)
            ]
        )
        return preprocessor