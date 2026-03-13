# MLflow Iris Classification Pipeline

This project builds, evaluates, tracks, and registers two classifiers on the Iris dataset using MLflow:

- Logistic Regression
- Random Forest

It is structured as a reproducible MLOps-style pipeline with experiment tracking, nested runs, artifact logging, payload validation, and model registry integration.

---

## 1) Workflow Explanation

The full pipeline is orchestrated in [main.py](main.py).

### Step-by-step flow

1. **MLflow setup**
	- Sets tracking server URI and experiment name from [src/config.py](src/config.py).

2. **Load and split data**
	- Loads Iris data and performs train/test split in [src/data.py](src/data.py).

3. **Start parent run**
	- A top-level run (`Main Run`) tracks the complete pipeline execution.

4. **Train child run: Logistic Regression**
	- Logs hyperparameters (`C`, `solver`, `max_iter`).
	- Trains model in [src/train.py](src/train.py).
	- Logs model artifact with MLflow (`lr_model`).
	- Validates sample inference payload via [src/validation.py](src/validation.py).
	- Evaluates metrics (accuracy, weighted F1) via [src/evaluate.py](src/evaluate.py).
	- Generates and logs confusion matrix via [src/plots.py](src/plots.py).
	- Registers the trained model version using [src/register.py](src/register.py).

5. **Train child run: Random Forest**
	- Logs hyperparameters (`n_estimators`, `max_depth`, `random_state`).
	- Repeats the same train → validate → evaluate → plot → register flow.

6. **Comparison artifacts**
	- Logs bar-chart comparison artifacts for accuracy and F1.

---

## 2) Which Model Performed Best?

Based on your latest run logs:

- Logistic Regression: Accuracy = **1.0000**, Weighted F1 = **1.0000**
- Random Forest: Accuracy = **1.0000**, Weighted F1 = **1.0000**

### Conclusion

Both models are tied on current evaluation metrics. There is **no single winner** by accuracy/F1 in this run.

If you need one production default, choose:

- **Logistic Regression** for simplicity and interpretability, or
- **Random Forest** if you expect more non-linear behavior on future data.

---

## 3) Project Structure

- [main.py](main.py): pipeline orchestration
- [src/config.py](src/config.py): constants and config values
- [src/data.py](src/data.py): dataset loading and split
- [src/train.py](src/train.py): model training helpers
- [src/evaluate.py](src/evaluate.py): metric computation
- [src/validation.py](src/validation.py): payload/schema validation
- [src/plots.py](src/plots.py): confusion matrix and comparison plots
- [src/register.py](src/register.py): model registration helper

---

## 4) How to Run

1. Install dependencies:

	pip install -r requirements.txt

2. Start MLflow tracking server (matching `TRACKING_URI`):

	mlflow server --host 0.0.0.0 --port 5000

3. Run the pipeline:

	python3 main.py

---

## 5) Outputs and Artifacts

- MLflow experiment with nested child runs
- Logged parameters and metrics for both models
- Confusion matrix images under `artifacts/`
- Model comparison plot under `artifacts/`
- Registered model versions:
  - `MLflow_Iris_Classifier_LogisticRegression`
  - `MLflow_Iris_Classifier_RandomForest`

---

## 6) Notes

- The payload validator enforces:
  - 2D input
  - exactly 4 features
  - no missing values
- Existing registered model names create new versions automatically on re-run.
