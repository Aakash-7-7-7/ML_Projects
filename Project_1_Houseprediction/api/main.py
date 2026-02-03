from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np 
import joblib
import os 


app=FastAPI(title="House Price Prediction API")

MODEL_URL="models:/HousePriceModel@production"
model=mlflow.sklearn.load_model(MODEL_URL)
scaler=joblib.load('scaler.pkl')

class HouseINput(BaseModel):
    #Gr Liv Area','Year Built

    grliv:int
    year:int

@app.get('/')
def home():
    return{'message':"House Price Prediction Running"}

@app.post('/predict')

def predict(data:HouseINput):
    input_data=np.array([[data.grliv,data.year]])

    scaled_input=scaler.transform(input_data)
    prediction=model.predict(input_data)

    return{"predicted_price":round(float(prediction[0]))}
