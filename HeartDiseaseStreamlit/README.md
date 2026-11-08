# Heart Disease Risk Prediction App

A Streamlit application that uses a saved K-Nearest Neighbors (KNN) model to demonstrate heart disease risk prediction from user-entered health measurements. The repository includes the app, notebook, dataset, scaler, expected feature names, and trained model.

> **Important:** This is an educational machine learning project, not a medical device. Predictions are not a diagnosis and should not guide treatment decisions.

## Repository files

- `app.py` — Streamlit user interface and prediction logic.
- `heart.ipynb` — notebook with dataset exploration, preprocessing, and model training.
- `heart.csv` — dataset used in the notebook.
- `KNN_heart.pkl` — serialized trained KNN model.
- `scaler.pkl` — fitted feature scaler.
- `Expected_columns.pkl` — expected feature order used by the app.
- `requirements.txt` — Python dependency versions.

Keep the three `.pkl` files beside `app.py`; the app loads them using relative paths.

## Run locally

1. Install Python 3.10 or a compatible version.
2. Open a terminal in this project folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

## Deploy with Streamlit Community Cloud

1. Upload all files in this folder to the root of a GitHub repository.
2. In Streamlit Community Cloud, choose **Create app** and connect that repository.
3. Select the branch containing the files and set the app file path to `app.py`.
4. Deploy.

The model files must be committed to the repository for this app's current loading approach. Do not commit secrets or credentials.

## Notes

The app uses the saved scaler and feature-column order so inputs match the model's training representation. Serialized scikit-learn models can be sensitive to library-version changes; if loading fails in deployment, retrain and export the model artifacts using the versions in `requirements.txt` (or pin the matching versions used to create the artifacts).
