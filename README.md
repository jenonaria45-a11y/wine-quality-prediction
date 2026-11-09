# Wine Quality Prediction System

## Project Description
The Wine Quality Prediction System is a machine learning project that predicts wine quality using physicochemical characteristics. It uses a Random Forest Classifier and a Streamlit dashboard.

## Objectives
- Predict wine quality based on its chemical properties.
- Classify wine samples into GOOD QUALITY and NEEDS IMPROVEMENT.
- Provide an easy-to-use dashboard for CSV uploads and predictions.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib
- Google Colab
- GitHub

## Dataset
The project uses the UCI Wine Quality Red Wine Dataset containing 1,599 records.

The original `quality` column is excluded from the model input to prevent target leakage. The target is converted into two classes:
- GOOD QUALITY: quality score >= 6
- NEEDS IMPROVEMENT: quality score < 6

## Machine Learning Model
Random Forest Classifier with:
- 150 estimators
- Random state: 42
- Balanced class weights
- Median imputation
- Standard scaling

## Project Files
- `app.py` – Streamlit application
- `requirements.txt` – Required libraries
- `wine_quality.csv` – Wine dataset
- `wine_model.pkl` – Trained machine learning model

## How to Run
1. Install the required libraries:
   `pip install -r requirements.txt`
2. Run the application:
   `streamlit run app.py`
3. Upload a CSV file containing the required wine predictor columns.
4. View and download the prediction results.

## Prediction Categories
- GOOD QUALITY
- NEEDS IMPROVEMENT

## Important Note
This project is an educational prototype. Predictions are preliminary and should not replace professional wine-quality evaluation.
