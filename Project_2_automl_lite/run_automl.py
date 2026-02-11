'''import os
from automl.data_loader import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "housing.csv")

loader = DataLoader(DATA_PATH)
df = loader.load_data()

print(df.isnull().sum())


from automl.data_loader import DataLoader
from automl.preprocessor import Preprocessor

loader=DataLoader('data/housing.csv')
df=loader.load_data()

preprocessor=Preprocessor(target_column='median_house_value')

X,y=preprocessor.split_feature_target(df)
X_train,X_test,y_train,y_test=preprocessor.train_test_split(X,y)

pipeline=preprocessor.build_pipeline(X_train)

print("Success")
'''

from automl.models import SupportVectorModel

model = SupportVectorModel()
print(model.get_name())
