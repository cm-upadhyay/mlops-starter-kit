import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import json
import os

def train_model():
    print("Starting model training...")
    df = pd.read_csv(os.path.join('data', 'iris.csv'))

    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['species']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = LogisticRegression(max_iter=200, solver='liblinear')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}")

    metrics = {"accuracy": accuracy}

    # Save the model and metrics locally
    joblib.dump(model, 'iris_model.pkl')
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)
    
    print("Model and metrics saved locally.")

if __name__ == "__main__":
    train_model()