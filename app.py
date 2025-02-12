from data import preproc_data
from model import load_model
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd


app = FastAPI()

model = load_model("model.pkl")
if model is None:
    raise ValueError("❌ No model found in GCS. Make sure the model is trained and saved!")


class InputData(BaseModel):
    HomePlanet: Optional[str] = None 
    CryoSleep: Optional[bool] = None
    Destination: Optional[str] = None 
    Age: Optional[float] = None 
    VIP: Optional[bool] = None 
    RoomService: Optional[float] = None 
    FoodCourt: Optional[float] = None 
    ShoppingMall: Optional[float] = None 
    Spa: Optional[float] = None 
    VRDeck: Optional[float] = None 


@app.get("/")
def home():
    return {"message": "Titanic Survival Prediction API is running!"}

@app.post("/predict")
def predict(data: InputData):
    # Convert input data to DataFrame (matching training format)
    input_df = pd.DataFrame([data.dict()])
    preproced_data = preproc_data(input_df, training=False)
    prediction = model.predict(preproced_data)

    return {'prediction': prediction[0]}


