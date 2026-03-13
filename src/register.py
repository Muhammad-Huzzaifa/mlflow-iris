import mlflow

def register_model(run_id, artifact_path, model_name):

    model_uri = f"runs:/{run_id}/{artifact_path}"

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    return model_version
