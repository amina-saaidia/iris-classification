"""Predict the species of a new iris sample using the saved model."""
import argparse
import joblib
import pandas as pd

def predict(sepal_length, sepal_width, petal_length, petal_width):
    model = joblib.load("models/best_model.pkl")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--sepal_length", type=float, required=True)
    parser.add_argument("--sepal_width", type=float, required=True)
    parser.add_argument("--petal_length", type=float, required=True)
    parser.add_argument("--petal_width", type=float, required=True)
    args = parser.parse_args()

    result = predict(args.sepal_length, args.sepal_width, args.petal_length, args.petal_width)
    print(f"Predicted species: {result}")