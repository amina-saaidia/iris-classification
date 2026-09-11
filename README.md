# 🌸 EXPS - Iris Flower Classification

Classifies iris flowers into 3 species (Setosa, Versicolor, Virginica) from
sepal and petal measurements, built as Task 1 of the EXPS Nexus Data Science
Internship (Algeria Pilot Cohort 01).

## Overview

**Task requirements covered:**
- A **multi-class classification** model (not regression) predicting flower
  species from 4 physical measurements
- Full **data cleaning**, **feature scaling**, and **model comparison**
- Built entirely with **Pandas**, **Scikit-learn**, and **Matplotlib/Seaborn**

## Dataset

- **Source:** scikit-learn's built-in Iris dataset (Fisher, 1936)
- 150 original samples, **149 rows after removing 1 duplicate**
- 4 features: sepal length, sepal width, petal length, petal width (cm)
- Target classes: Setosa (50), Versicolor (50), Virginica (49)

## Methodology

1. **Cleaning & preprocessing** (`src/data_prep.py`)
   - Loaded the dataset, dropped 1 duplicate row, verified zero missing values
   - Stratified 80/20 train/test split to preserve class balance

2. **EDA** (`notebooks/EDA.ipynb`)
   - Pairplot and correlation heatmap across all 4 features
   - Found Setosa fully separable by petal measurements alone, and a strong
     (r = 0.96) correlation between petal length and petal width

3. **Modeling** (`src/train.py`)
   - Three classifiers compared, each wrapped in a Scikit-learn `Pipeline`:
     **Logistic Regression**, **K-Nearest Neighbors** (K=5), **Random Forest**
   - `StandardScaler` applied for Logistic Regression and KNN, justified by
     the feature-scale differences found in EDA

4. **Evaluation**
   - Accuracy, macro Precision, macro Recall, macro F1, and confusion
     matrices computed on the held-out test set for each model

### A note on a model-selection bug

An earlier version of `train.py` selected the "best" model using a strict
`if accuracy > best_score` comparison. Since all three models tied exactly
on accuracy (0.9333), that comparison never triggered past the first model
evaluated — so the model saved as `best_model.pkl` was decided by dictionary
order, not real performance. This is now fixed to select on **macro
precision** instead, which correctly identifies **KNN (K=5)** as the
strongest model.

## Results

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1 Score (Macro) |
| --- | --- | --- | --- | --- |
| **KNN (K=5)** | 0.9333 | **0.9444** | 0.9333 | 0.9327 |
| Logistic Regression | 0.9333 | 0.9333 | 0.9333 | 0.9333 |
| Random Forest | 0.9333 | 0.9333 | 0.9333 | 0.9333 |

**Best model saved:** KNN (K=5) — saved as a pipeline artifact to
`models/best_model.pkl`.

**KNN confusion matrix** (rows = actual, columns = predicted):

```
                Predicted Setosa   Predicted Versicolor   Predicted Virginica
Actual Setosa           10                  0                     0
Actual Versicolor        0                 10                     0
Actual Virginica          0                  2                     8
```

## Key Findings

- All three models tied on accuracy — a close result explained by the EDA:
  Setosa is fully separable, so the only real difficulty is the slight
  overlap between Versicolor and Virginica.
- **Setosa is completely linearly separable** — 10/10 correct across every
  model tested.
- Every model's only errors fall between **Versicolor and Virginica**,
  matching the overlap identified during EDA.
- KNN kept those errors to just 2 misclassifications (the fewest of the
  three), which is what earns it the top precision score.
- Feature scaling materially affects KNN and Logistic Regression, but not
  Random Forest, which is scale-invariant by construction.

## Live Demo

Try the interactive Streamlit app:
**https://iris-classification-3brcqg4jkei9zpijybzxaz.streamlit.app/**

## How to Run Locally

```bash
git clone https://github.com/amina-saaidia/EXPS_Iris_Flower_Classification.git
cd EXPS_Iris_Flower_Classification
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train and evaluate models
python src/train.py

# Predict on a new sample
python src/predict.py --sepal_length 5.1 --sepal_width 3.5 --petal_length 1.4 --petal_width 0.2

# Or launch the interactive app
streamlit run app.py

# Or open the full walkthrough
jupyter notebook notebooks/EDA.ipynb
```

## Project Structure

```text
EXPS_Iris_Flower_Classification/
├── .streamlit/
│   └── config.toml
├── data/
│   ├── iris_raw.csv
│   └── iris_cleaned.csv
├── models/
│   └── best_model.pkl
├── notebooks/
│   └── EDA.ipynb
├── presentation/
│   └── capstone_presentation.pdf
├── src/
│   ├── data_prep.py
│   ├── predict.py
│   └── train.py
├── .gitignore
├── app.py
├── README.md
└── requirements.txt
```

## Real-World Applications

- **Botanical / agricultural research** — quick species identification from
  simple physical measurements, without needing an expert taxonomist on hand.
- **Educational tool** — a classic, well-understood example for teaching
  classification workflows end to end.
- **Reusable pipeline template** — the same clean → scale → compare → select
  workflow used here generalizes directly to other small tabular
  classification problems.
- **Lightweight field tools** — a model this small and fast (KNN, 4 input
  features) could realistically power a simple on-device identification aid.

## Scope and Limitations

- Trained on a small, clean, well-known benchmark dataset (149 samples) —
  real-world flower identification (e.g. from photos, in varied field
  conditions) involves far more noise and species variety than this model
  has ever seen.
- Only 4 numeric measurements are used; there are no image-based or
  environmental features.
- Results are specific to this dataset's 3 species and measurement
  conventions, and shouldn't be assumed to generalize to other iris
  varieties or measurement setups.

## Future Improvements
- Replace the single train/test split with k-fold cross-validation for more robust metrics
- Tune hyperparameters (e.g. KNN's K, Gradient Boosting's learning rate/depth) via grid search
- Deeper error analysis — which specific cases each model gets wrong, and why
- Evaluate on a held-out real-world sample beyond this benchmark dataset

## Author

Built as Task 1 of the EXPS Nexus Data Science Internship (Algeria Pilot
Cohort 01).
