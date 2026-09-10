
# EXPS Iris Flower Classification

Classifies iris flowers into 3 species (setosa, versicolor, virginica) based on sepal and petal measurements using Scikit-Learn. Developed as part of the **EXPS Nexus Data Science Internship Program (Task 1)**[cite: 4].

## Dataset

The classic Iris dataset: 150 original samples, 4 features (sepal length/width, petal length/width), and 3 balanced classes (50 samples each)[cite: 4]. Duplicate entries were cleaned during preprocessing to prevent data leakage between training and testing splits.

## Setup

```bash
git clone [https://github.com/amina-saaidia/EXPS_Iris_Flower_Classification.git](https://github.com/amina-saaidia/EXPS_Iris_Flower_Classification.git)
cd EXPS_Iris_Flower_Classification
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt

```

## Usage

Train and evaluate the models:

```bash
python src/train.py

```

Predict on a new sample:

```bash
python src/predict.py --sepal_length 5.1 --sepal_width 3.5 --petal_length 1.4 --petal_width 0.2

```

## Approach & Model Choices

Three classification algorithms were compared within Scikit-Learn `Pipeline` architectures to ensure proper feature scaling (`StandardScaler`) without data leakage:

* **Logistic Regression** — A scalable linear model wrapped with feature standardization, serving as a clean baseline for multi-class decision boundaries.
* **K-Nearest Neighbors (KNN)** — A distance-based classifier ($K=5$) requiring feature standardization to ensure all physical measurements contribute equally to distance calculations.
* **Random Forest** — An ensemble tree method included to evaluate non-linear feature interactions without requiring distance normalization.

Data was deduplicated and split 80/20 (train/test) using stratification to preserve class balance. Models were evaluated on unseen test data using Accuracy, Macro Precision, Macro Recall, and Macro F1 Score.

## Results

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1 Score (Macro) |
| --- | --- | --- | --- | --- |
| **Logistic Regression** | **0.9333** | 0.9333 | **0.9333** | **0.9333** |
| **Random Forest** | **0.9333** | 0.9333 | **0.9333** | **0.9333** |
| **KNN (K=5)** | **0.9333** | **0.9444** | **0.9333** | 0.9327 |

Best model saved: **KNN (K=5)** — saved as a unified pipeline artifact to `models/best_model.pkl`.

## Key Insights

* **Deduplication & Realistic Baselines:** Cleaning duplicate entries prior to the stratified split eliminated artificial sample memorization, establishing a realistic ~93.3% performance baseline.
* **Feature Standardization Impact:** Wrapping KNN in a `StandardScaler` pipeline ensured that sepal and petal features contributed proportionally to distance calculations.
* **Class Separability:** Setosa remains completely linearly separable (10/10 test samples correct across all models), while minor overlap exists between Versicolor and Virginica boundary samples.

## Conclusion

Both Logistic Regression and Random Forest achieved strong performance across all primary metrics (93.3% Accuracy and F1 Score) on unseen test data. Logistic Regression was selected as the saved artifact due to its parameter efficiency, rapid inference speed, and balanced error distribution across overlapping classes.

This project demonstrates a production-ready Machine Learning workflow: data preprocessing, feature scaling, leak-free model evaluation, automated artifact serialization, and CLI inference execution.

## Project Structure

```text
Iris_Classification/
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

## Author

Built as Task 3 of the EXPS Nexus Data Science Internship.
