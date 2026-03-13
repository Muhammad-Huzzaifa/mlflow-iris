import mlflow
import mlflow.sklearn

from src.config import EXPERIMENT_NAME, TRACKING_URI, MODEL_NAME
from src.data import load_data
from src.train import train_logistic_regression, train_random_forest
from src.evaluate import evaluate_model
from src.plots import plot_confusion_matrix, plot_model_comparison
from src.validation import validate_payload
from src.register import register_model

def run_pipeline():

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run(run_name="Main Run") as run:

        print("Run ID:", run.info.run_id)

        lr_acc, rf_acc = None, None
        lr_f1, rf_f1 = None, None

        with mlflow.start_run(run_name="Logistic Regression", nested=True) as lr_run:

            lr_params = {
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 1000,
            }
            mlflow.log_params(lr_params)

            lr_model = train_logistic_regression(X_train, y_train, **lr_params)
            mlflow.sklearn.log_model(
                sk_model=lr_model,
                name="lr_model",
                serialization_format="skops",
                pip_requirements=["scikit-learn", "skops"],
            )

            lr_payload = validate_payload(X_test[:5])
            print(f"Payload: {lr_payload} Output: {lr_model.predict(lr_payload)}")

            lr_acc, lr_f1, lr_preds = evaluate_model(lr_model, X_test, y_test)
            mlflow.log_metrics({"lr_acc": lr_acc, "lr_f1": lr_f1})
            print(f"Logistic Regression - Accuracy: {lr_acc:.4f}, F1 Score: {lr_f1:.4f}")

            lr_cm = plot_confusion_matrix(y_test, lr_preds, "logistic_regression")
            mlflow.log_artifact(lr_cm)

            lr_run_id = lr_run.info.run_id
            lr_version = register_model(lr_run_id, "lr_model", f"{MODEL_NAME}_LogisticRegression")
            print(f"Logistic Regression Version: {lr_version.version}")

        with mlflow.start_run(run_name="Random Forest", nested=True) as rf_run:

            rf_params = {
                "n_estimators": 100,
                "max_depth": 5,
                "random_state": 42
            }
            mlflow.log_params(rf_params)

            rf_model = train_random_forest(X_train, y_train, **rf_params)
            mlflow.sklearn.log_model(
                sk_model=rf_model,
                name="rf_model",
                serialization_format="skops",
                pip_requirements=["scikit-learn", "skops"],
            )

            rf_payload = validate_payload(X_test[:5])
            print(f"Payload: {rf_payload} Output: {rf_model.predict(rf_payload)}")

            rf_acc, rf_f1, rf_preds = evaluate_model(rf_model, X_test, y_test)
            mlflow.log_metrics({"rf_acc": rf_acc, "rf_f1": rf_f1})
            print(f"Random Forest - Accuracy: {rf_acc:.4f}, F1 Score: {rf_f1:.4f}")

            rf_cm = plot_confusion_matrix(y_test, rf_preds, "random_forest")
            mlflow.log_artifact(rf_cm)

            rf_run_id = rf_run.info.run_id
            rf_version = register_model(rf_run_id, "rf_model", f"{MODEL_NAME}_RandomForest")
            print(f"Random Forest Version: {rf_version.version}")

        acc_metrics = {
            "Logistic Regression": lr_acc,
            "Random Forest": rf_acc
        }
        acc_comparison_path = plot_model_comparison(acc_metrics)
        mlflow.log_artifact(acc_comparison_path)

        f1_metrics = {
            "Logistic Regression": lr_f1,
            "Random Forest": rf_f1
        }
        f1_comparison_path = plot_model_comparison(f1_metrics)
        mlflow.log_artifact(f1_comparison_path)

if __name__ == "__main__":
    run_pipeline()
