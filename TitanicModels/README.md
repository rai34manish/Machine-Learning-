# Titanic Survival Prediction

A beginner-friendly machine learning project comparing five classifiers on Seaborn's Titanic passenger dataset.

## Project files

- `TitanicModels.ipynb` — original notebook with exploration, model training, and evaluation.
- `titanic_models.py` — runnable script that trains the five models and prints accuracy, confusion matrices, classification reports, and a sorted accuracy summary.

## Results from the notebook

| Model | Test accuracy |
|---|---:|
| Logistic Regression | 80.34% |
| K-Nearest Neighbors (k=5) | 79.21% |
| Gaussian Naive Bayes | 77.53% |
| Decision Tree | 80.34% |
| Support Vector Machine (RBF) | **81.46%** |

The notebook uses an 80/20 train-test split (`random_state=42`), giving 178 test examples. These are baseline scores from a single split, not cross-validated or tuned results.

## Setup

Requires Python 3. Install dependencies with:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn jupyter
```

## Run

From this folder, run the Python script:

```bash
python titanic_models.py
```

Or open `TitanicModels.ipynb` in Jupyter Notebook or Google Colab and run its cells. The dataset is loaded from Seaborn and may need to be downloaded on first use.

## Workflow

1. Load the Seaborn Titanic dataset.
2. Drop selected redundant or high-missingness columns; fill missing ages with the mean and remove rows missing embarkation information.
3. Encode sex and embarkation values, then split predictors from the survival target.
4. Split into training and test sets.
5. Train Logistic Regression, K-Nearest Neighbors, Gaussian Naive Bayes, Decision Tree, and an RBF Support Vector Machine.
6. Compare accuracy, confusion matrices, and classification reports.

## Limitations and next steps

The notebook performs some data cleaning and category encoding before splitting into train and test sets, which can make evaluation optimistic. The Python script keeps the same general workflow for a like-for-like baseline. A stronger version would put preprocessing inside scikit-learn pipelines, use stratified cross-validation, and tune parameters using training data only.

## License

No license is included. Add a license if you want to specify reuse and redistribution terms.
