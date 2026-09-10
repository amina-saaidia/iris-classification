"""Predict the species of a new iris sample using the saved model pipeline."""
import os
import argparse
import joblib
import pandas as pd

# Dynamic path resolution to locate best_model.pkl robustly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models") if not os.path.basename(BASE_DIR) == "src" else os.path.join(BASE_DIR, "..", "models")
MODEL_PATH = os.path.normpath(os.path.join(MODEL_DIR, "best_model.pkl"))

def predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float) -> str:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run train.py first.")

    model = joblib.load(MODEL_PATH)
    species_names = ["setosa", "versicolor", "virginica"]

    sample = pd.DataFrame([{
        "sepal length (cm)": sepal_length,
        "sepal width (cm)": sepal_width,
        "petal length (cm)": petal_length,
        "petal width (cm)": petal_width,
    }])

    pred = model.predict(sample)[0]
    return species_names[pred]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iris Species Predictor")
    parser.add_argument("--sepal_length", type=float, required=True)
    parser.add_argument("--sepal_width", type=float, required=True)
    parser.add_argument("--petal_length", type=float, required=True)
    parser.add_argument("--petal_width", type=float, required=True)
    args = parser.parse_args()

    result = predict(args.sepal_length, args.sepal_width, args.petal_length, args.petal_width)
    print(f"Predicted species: {result}")