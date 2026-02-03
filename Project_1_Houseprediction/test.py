from src.dataloader import DataLoader
from src.preprocessor import Preprocessor
from src.model import Housepred
from src.trainer import Trainer

loader=DataLoader('data/AmesHousing.csv')
data=loader.load_data()

pre=Preprocessor()
X,y=pre.split_data(data)
X_train,X_test,y_train,y_test=pre.train_test_split(X,y)
X_trained_scaled,X_test_scaled=pre.scale_data(X_train,X_test)

model=Housepred()
trainer=Trainer(model,pre.scaler)

metrics=trainer.train_and_log(
    X_trained_scaled,
    y_train,
    X_test_scaled,
    y_test
)
print(metrics)

print("Preprocessing Succesful")