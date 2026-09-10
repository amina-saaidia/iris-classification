"""Load, clean, export, and prepare the Iris dataset."""
import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Resolve paths relative to the project root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))
CLEAN_DATA_PATH = os.path.join(DATA_DIR, "iris_cleaned.csv")

def load_and_clean_data() -> pd.DataFrame:
    """Load Iris dataset from scikit-learn, remove duplicates, and export iris_cleaned.csv."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map(dict(enumerate(iris.target_names)))

    # Data cleaning: remove duplicate records
    df_cleaned = df.drop_duplicates().reset_index(drop=True)

    # Save cleaned data CSV to data/ folder for transparency
    os.makedirs(DATA_DIR, exist_ok=True)
    df_cleaned.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Cleaned dataset exported to: {CLEAN_DATA_PATH} ({len(df_cleaned)} rows)")

    return df_cleaned

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split clean dataset into train and test sets using stratification."""
    X = df.drop(columns=['species', 'species_name'])
    y = df['species']
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

if __name__ == "__main__":
    df = load_and_clean_data()
    print("\nCleaned Data Sample:")
    print(df.head())