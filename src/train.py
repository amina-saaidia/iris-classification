"""Train and compare multiple models on the Iris dataset."""
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from data_prep import load_and_clean_data, split_data

# Dynamic path resolution: finds the root directory relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# If train.py is inside a src/ folder, change '..' to target the project root models folder
MODEL_DIR = os.path.join(BASE_DIR, "models") if not os.path.basename(BASE_DIR) == "src" else os.path.join(BASE_DIR, "..", "models")
MODEL_PATH = os.path.normpath(os.path.join(MODEL_DIR, "best_model.pkl"))

def train_and_evaluate():
    df = load_and_clean_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # Models wrapped in pipelines where feature scaling is needed
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5))
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(random_state=42))
        ]),
    }

    results = {}
    best_model = None
    best_score = 0.0
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='macro')
        rec = recall_score(y_test, preds, average='macro')
        f1 = f1_score(y_test, preds, average='macro')

        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

        print(f"\n{name}")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Confusion Matrix:\n{confusion_matrix(y_test, preds)}")

        if prec > best_score:
            best_score = prec
            best_model = model
            best_name = name

    print(f"\n>>> Best model: {best_name} (precision: {best_score:.4f})")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved best model pipeline to: {MODEL_PATH}")

    return results

if __name__ == "__main__":
    train_and_evaluate()