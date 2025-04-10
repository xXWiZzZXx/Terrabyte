import os
import json
import pandas as pd
import mlflow.sklearn

def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model")
    model = mlflow.sklearn.load_model(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        inputs = pd.DataFrame.from_dict(data)
        preds = model.predict(inputs)
        return preds.tolist()
    except Exception as e:
        return {"error": str(e)}
