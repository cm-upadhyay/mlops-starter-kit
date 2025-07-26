import pytest
import pandas as pd
import os
import subprocess
import joblib
import json

@pytest.fixture(scope="module")
def training_run_artifacts():
    """
    Runs train.py to produce a model and metrics, then loads them.
    """
    # Clean up any existing artifacts before running train.py
    if os.path.exists('iris_model.pkl'):
        os.remove('iris_model.pkl')
    if os.path.exists('metrics.json'):
        os.remove('metrics.json')

    # Run train.py as a subprocess
    result = subprocess.run(["python", "train.py"], capture_output=True, text=True, check=True)
    print("train.py stdout:\n", result.stdout)

    # Load the generated model and metrics from local files
    model = joblib.load('iris_model.pkl')
    with open('metrics.json', 'r') as f:
        metrics = json.load(f)

    return model, metrics

def test_model_accuracy_above_threshold(training_run_artifacts):
    model, metrics = training_run_artifacts
    accuracy = metrics.get("accuracy")
    assert accuracy is not None, "Accuracy not found in metrics.json."
    assert accuracy >= 0.80, f"Model accuracy {accuracy:.4f} is below the threshold of 0.80."

def test_log_metrics(training_run_artifacts):
    model, metrics = training_run_artifacts
    print("\n--- Model Metrics ---")
    print(json.dumps(metrics, indent=4))
    print("---------------------")
    assert True