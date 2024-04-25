import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

def load_data():
    # Load the dataset
    data = pd.read_csv("dataset_web.csv")
    return data

def preprocess_data(data):
    # Extract numeric value from 'Label' column and store it in a new column 'Threat'
    data['Threat'] = data['Label'].str.extract('(\d+)').astype(float)

    # Drop unnecessary columns and the 'Threat' column if NaN values are present
    data_cleaned = data.drop(['Label', 'Timestamp'], axis=1)
    if data_cleaned['Threat'].isnull().any():
        data_cleaned = data_cleaned.dropna(subset=['Threat'])

    return data_cleaned

def train_model(X, y):
    # Define the columns to be one-hot encoded
    categorical_cols = ['Request Method', 'Request Path', 'Request Parameters', 'User-Agent',
                        'Referrer', 'IP Address', 'Content-Type', 'Response Code']

    # Define a ColumnTransformer to apply transformations to specific columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ], remainder='passthrough')

    # Create the model pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier())])

    # Fit the model
    model.fit(X, y)

    return model

def load_model(X_train, y_train):
    # Load the trained model
    model = train_model(X_train, y_train)
    return model

def predict_threat(model, input_data):
    # Prepare input data for prediction
    X_input = pd.DataFrame([input_data])

    # Make prediction
    threat_status = model.predict(X_input)

    return threat_status
