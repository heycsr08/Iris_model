import json
import joblib
import numpy as np
import os

# Called once when the service starts
def init():
    global model
    
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.pkl")
    model = joblib.load(model_path)

# Called for each request
def run(raw_data):
    try:
        # Convert input JSON to numpy array
        data = json.loads(raw_data)
        input_data = np.array(data["data"])
        
        # Prediction
        predictions = model.predict(input_data)
        
        # Optional: convert to list for JSON serialization
        return {
            "predictions": predictions.tolist()
        }

    except Exception as e:
        return {
            "error": str(e)
        }