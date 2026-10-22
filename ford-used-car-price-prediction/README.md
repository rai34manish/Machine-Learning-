# Ford Used Car Price Prediction

A beginner machine learning project that explores Ford used-car listings and predicts their prices with linear regression.

## Project files

- `linearregression.ipynb` — the notebook with data exploration, charts, preprocessing, model training, and evaluation.
- `ford.csv` — the dataset used by the notebook (17,966 rows and 9 columns).

The notebook reads `ford.csv` from the same folder, so keep both files together.

## Dataset

The dataset contains Ford car listing details. The target column is `price`; the input columns are `model`, `year`, `transmission`, `mileage`, `fuelType`, `tax`, `mpg`, and `engineSize`. The notebook checks for missing values and explores relationships between these fields and price.

Dataset source: [Ford Car Price Prediction on Kaggle](https://www.kaggle.com/datasets/adhurimquku/ford-car-price-prediction)

Please check the Kaggle dataset page for its current license and terms before redistributing the CSV. The dataset is provided by its Kaggle uploader; this project does not claim ownership of it.

## What the notebook does

1. Loads and explores the data, including summary statistics and missing values.
2. Creates charts for price and the main vehicle features.
3. Prepares two versions of the input data: one-hot encoding and label encoding.
4. Scales numerical features and splits the data into training and test sets.
5. Trains linear regression models and reports R² and adjusted R² scores.

The saved notebook shows an R² score of about **0.840** for one-hot encoding and **0.746** for label encoding. Results can vary if the data or notebook changes. These scores are a starting point, not a guarantee of prediction accuracy on future listings.

## Run the notebook on your computer

You need Python 3 and the packages below. Open PowerShell or a terminal in this project folder and install them:

```bash
python -m pip install numpy pandas seaborn matplotlib scikit-learn jupyter
```

Then start Jupyter:

```bash
jupyter notebook linearregression.ipynb
```

Run the cells from top to bottom. If `jupyter` is not recognized, try:

```bash
python -m notebook linearregression.ipynb
```

## Run it on Kaggle

Open the [Kaggle notebook](https://www.kaggle.com/code/rai34manish/linearregression/edit) and attach the Ford dataset to the notebook. The GitHub copy reads the local `ford.csv` file; the Kaggle copy may use Kaggle's input-file path instead.

## Notes

- This is an educational project and is not financial or vehicle valuation advice.
- No license is specified here for the project code. Add a license only if you decide how you want others to reuse your code.
