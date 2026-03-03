from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os
from starter.ml.data import process_data
from starter.ml.model import inference

# Get Model
model_path = os.path.join(os.getcwd(), "model")

# Load Artifacts

model = joblib.load(os.path.join(model_path, "model.pkl"))
encoder = joblib.load(os.path.join(model_path, "encoder.pkl"))
lb = joblib.load(os.path.join(model_path, "label_binarizer.pkl"))
cat_features = joblib.load(os.path.join(model_path, "cat_features.pkl"))

# Initialize FastAPI
app = FastAPI()

class IncomeDate(BaseModel):
    age: int
    workclass: str
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "age": 37,
                "workclass": "Self-emp-not-inc",
                "education": "Masters",
                "education-num": 7,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 7000,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "Germany"
            }
        }

# Root Get Endpoint
@app.get("/")
async def welcome():
    return {"message": "Welcome to Census Income Prediction API"}

# Prediction Post Endpoint
@app.post("/predict")
async def predict(data: IncomeDate):
    # Convert Pydantic object to DataFrame
    df = pd.DataFrame([data.model_dump(by_alias=True)])

    # Process Data (using artifacts)
    data, _, _, _ = process_data(
        df, 
        categorical_features=cat_features, 
        training=False, 
        encoder=encoder, 
        lb=lb
    )

    # Get Numerical Prediction
    preds = inference(model, data)

    # Convert back to String Label
    prediction = lb.inverse_transform(preds)

    # Return Prediction
    return {"prediction": prediction[0]}
