"""Train and compare multiple models on the Iris dataset."""
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
from data_prep import load_data, split_data

def train_and_evaluate():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(random_state=42),
    }

    results = {}
    best_model = None
    best_score = 0
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

        if acc > best_score:
            best_score = acc
            best_model = model
            best_name = name

    print(f"\n>>> Best model: {best_name} (accuracy: {best_score:.4f})")

    os.makedirs("../models", exist_ok=True)
    joblib.dump(best_model, "../models/best_model.pkl")
    print("Saved to models/best_model.pkl")

    return results

if __name__ == "__main__":
    train_and_evaluate()