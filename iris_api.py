from fastapi import FastAPI 
import joblib
import numpy as np
from sklearn.datasets import load_iris

# Load the iris dataset 
iris = load_iris()
# Initialize FastAPI app 
app = FastAPI()

# Load the trained model
model = joblib.load('iris_data.joblib')

# Define FastAPI endpoints 
@app.get("/")
async def read_root():
    return {"message": "Welcome to the model API!"}

@app.post("/predict/")
async def predict_species(data: dict):
# Implement your prediction logic here using the loaded model 
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features) 
    class_name = iris.target_names[prediction][0] 
    return {"class": class_name}
