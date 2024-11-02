from argparse import ArgumentParser
import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import plotting
import preprocessing_trip_duration as preprocessing
from sklearn.linear_model import LinearRegression

"""
usage:
    python code/main.py --training_set PATH --test_set PATH --out PATH

for example:
    python code/main.py --training_set /cs/usr/gililior/training.csv --test_set /cs/usr/gililior/test.csv --out predictions/trip_duration_predictions.csv 

"""

# implement here your load,preprocess,train,predict,save functions (or any other design you choose)


def run(training_set, test_set, out):
    # Step 1: Load and preprocess the training and test sets
    logging.info("Loading and preprocessing data...")
    train_df = pd.read_csv(training_set, encoding="ISO-8859-8")
    test_df = pd.read_csv(test_set, encoding="ISO-8859-8")

    # Preprocess the training and test data
    preprocessed_train_df, y_train = preprocessing.preprocess_train(train_df)
    preprocessed_test_df, _, test_trip_ids = preprocessing.preprocess_test(test_df)

    # Optionally, apply one-hot encoding or scaling to handle categorical features and scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(preprocessed_train_df)
    X_test = scaler.transform(preprocessed_test_df)

    # Step 2: Choose a model (uncomment one at a time to test different models)
    # ========================================
    # Option 1: Linear Regression
    # ========================================
    # model = LinearRegression()
    # model_name = "Linear Regression"

    # ========================================
    # Option 2: XGBRegressor
    # ========================================
    # model = XGBRegressor(reg_alpha=0.1, reg_lambda=1.0, objective='reg:squarederror')
    # model_name = "XGBRegressor"

    # ========================================
    # Option 3: RandomForest Regressor
    # ========================================
    model = RandomForestRegressor()
    model_name = "RandomForest Regressor"

    # Step 3: Train the selected model
    print(f"Training model: {model_name}")
    model.fit(X_train, y_train)

    # Step 4: Make predictions on the test set
    logging.info("Predicting test data...")
    test_predictions = model.predict(X_test)

    # Step 5: Evaluate the model on the training data
    train_predictions = model.predict(X_train)
    mse = mean_squared_error(y_train, train_predictions)
    r2 = model.score(X_train, y_train)
    print(f"Training Mean Squared Error (MSE): {mse}")
    print(f"Training R² Score: {r2}")

    # Perform cross-validation for additional evaluation
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    print(f"Cross-validation scores (MSE): {-scores}")

    # Step 6: Save predictions to CSV
    predictions_df = pd.DataFrame({
        "trip_id_unique": test_trip_ids,
        "duration": test_predictions
    })
    predictions_df.to_csv(out, index=False, encoding="ISO-8859-8")

    # Call the plotting function from the plotting module
    # plotting.plot_important_features(model, preprocessed_train_df)
