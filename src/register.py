import mlflow
from src.config import MODEL_NAME

def register_model(run_id):
    
    mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=MODEL_NAME
    )
