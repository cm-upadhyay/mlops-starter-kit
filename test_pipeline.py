import pytest
import pandas as pd
import os
import subprocess
import mlflow
from mlflow.tracking import MlflowClient

# --- Fixtures ---

@pytest.fixture(scope="module")
def raw_iris_data():
    """Provides the raw IRIS dataset for data validation tests."""
    df = pd.read_csv(os.path.join('data', 'iris.csv'))
    return df

@pytest.fixture(scope="module")
def training_run():
    """
    Runs the main train.py script and yields the MLflow run object.
    This ensures that tests are run against a fresh, completed training execution.
    """
    # Set the MLflow Tracking URI from the environment variable for the test
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        pytest.fail("MLFLOW_TRACKING_URI environment variable not set.")
    
    mlflow.set_tracking_uri(tracking_uri)
    
    # Run train.py as a separate process
    result = subprocess.run(["python", "train.py"], text=True, check=True)
    assert result.returncode == 0, f"train.py failed with error:\n{result.stderr}"

    # Find the latest run in the experiment to test against
    client = MlflowClient()
    experiment = client.get_experiment_by_name("iris-classifier-training")
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
    
    assert len(runs) > 0, "No MLflow runs found after executing train.py"
    
    yield runs[0] # Yield the most recent run object

# --- Data Validation Tests ---

def test_data_has_expected_columns(raw_iris_data):
    expected_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    assert all(col in raw_iris_data.columns for col in expected_columns), "Data does not have expected columns."

def test_data_has_no_missing_values(raw_iris_data):
    assert not raw_iris_data.isnull().any().any(), "Dataset contains missing values."

# --- Model Evaluation Test (adapted for MLflow) ---

def test_model_accuracy_above_threshold(training_run):
    run, _ = training_run
    accuracy = run.data.metrics.get("accuracy")
    assert accuracy is not None, "Accuracy metric not found in MLflow run."
    assert accuracy >= 0.80, f"Model accuracy {accuracy:.4f} is below the threshold of 0.80."