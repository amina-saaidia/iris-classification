 # Iris Flower Classification

Classifies iris flowers into 3 species (setosa, versicolor, virginica) based on
sepal and petal measurements, using scikit-learn.

## Dataset

The classic Iris dataset: 150 samples, 4 features (sepal length/width, petal
length/width), 3 balanced classes (50 samples each).

## Setup

```bash
git clone https://github.com/amina-saaidia/iris-classification.git
cd iris-classification
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Train the models:

```bash
python src/train.py
```

Predict on a new sample:

```bash
python src/predict.py --sepal_length 5.1 --sepal_width 3.5 --petal_length 1.4 --petal_width 0.2
```

## Results

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.9667   | 0.9697    | 0.9667 | 0.9666   |
| **KNN**              | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| Random Forest       | 0.9000   | 0.9024    | 0.9000 | 0.8997   |

Best model: **KNN** — saved to `models/best_model.pkl`.

## Key Insights

## Key Insights
- Petal length and petal width are the most discriminative features between species.
- Setosa is linearly separable from the other two classes; versicolor and
  virginica overlap slightly.
- KNN achieved perfect accuracy on this test split, likely because Iris is a
  small, well-separated dataset — on a larger or noisier dataset the gap
  between models would likely be smaller.
	
## Project Structure

```text
iris-classification/
├── data/
├── models/
├── notebooks/
├── src/
│   ├── train.py
│   └── predict.py
├── requirements.txt
└── README.md
```
