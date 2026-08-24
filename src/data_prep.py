"""Load and prepare the Iris dataset."""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

def load_data():
    """Load iris dataset as a pandas DataFrame."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map(dict(enumerate(iris.target_names)))
    return df

def split_data(df, test_size=0.2, random_state=42):
    """Split into train/test sets."""
    X = df.drop(columns=['species', 'species_name'])
    y = df['species']
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nClass balance:\n{df['species_name'].value_counts()}")