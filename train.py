import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

# Set up MLflow Tracking
# The GitHub Actions workflow will set this environment variable
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("iris-classifier-training")

def train_model():
    print("Starting model training...")

    df = pd.read_csv(os.path.join('data', 'iris.csv'))

    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['species']

    with mlflow.start_run() as run:
        # Log hyperparameters
        mlflow.log_param("solver", "liblinear")
        mlflow.log_param("max_iter", 200)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        model = LogisticRegression(max_iter=200, solver='liblinear')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {accuracy}")
        mlflow.log_metric("accuracy", accuracy)

        # Log the model to the MLflow run and register it in the Model Registry
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="iris-classifier"
        )
        print(f"Model saved and registered as 'iris-classifier'. Run ID: {run.info.run_id}")

if __name__ == "__main__":
    train_model()