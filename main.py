from argparse import ArgumentParser
import logging
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import preprocessing
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotting
from sklearn.model_selection import cross_val_score


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True, help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True, help="path to the test set")
    parser.add_argument('--out', type=str, required=True,
                        help="path of the output file as required in the task description")
    args = parser.parse_args()

    # Step 1: Load and preprocess the training and test sets
    logging.info("Preprocessing train...")
    train = pd.read_csv(args.training_set, encoding="ISO-8859-8")
    test_df = pd.read_csv(args.test_set, encoding="ISO-8859-8")

    preprocessed_train_df = preprocessing.preprocess_data(train, is_train=True)
    preprocessed_test_df = preprocessing.preprocess_data(test_df, is_train=False)
    preprocessed_train_df.to_csv(
        "C:/Users/PC/Documents/Year3SemesterB/67577IML/Hackathon/data/HU.BER/preprocessed_train_df.csv",
        index=False
    )

    # Separate target variable 'passengers_up' and drop it from features
    y_train = preprocessed_train_df['passengers_up']
    X_train = preprocessed_train_df.drop(columns=['passengers_up'])

    # Handle categorical columns with one-hot encoding
    X_train = pd.get_dummies(X_train, columns=['alternative'], drop_first=True)
    X_test = pd.get_dummies(preprocessed_test_df, columns=['alternative'], drop_first=True)

    # Ensure test data has the same columns as training data
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Ensure there are no NaN values in the training data
    X_train = X_train.fillna(0)  # Fill any remaining NaN values with 0 (optional)
    y_train = y_train.fillna(0)  # Ensure y_train has no NaN values (optional)

    # Step 2: Choose one model at a time to experiment with
    # ========================================
    # Option 1: Linear Regression (non-negative constraint)
    # ========================================
    # Uncomment this block to use the Linear Regression model with a non-negative constraint
    # model = LinearRegression()
    # model_name = "Linear Regression (positive=True)"

    # ========================================
    # Option 2: Poisson Regression
    # ========================================
    # Uncomment this block to use Poisson Regression with scaling (good for count data)
    # scaler = StandardScaler()  # Scale features for Poisson Regression
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    # model = PoissonRegressor(alpha=1e-12, max_iter=1000)
    # model_name = "Poisson Regression"

    # ========================================
    # Option 3: XGBRegressor (basic)
    # ========================================
    # Uncomment this block to use the XGBoost Regressor model
    # model = XGBRegressor()
    # model_name = "XGBRegressor"

    # ========================================
    # Option 4: Log Transformation of Target
    # ========================================
    # Uncomment this block to apply log transformation to the target variable (prevents negative predictions)
    y_train = np.log1p(y_train)
    model = XGBRegressor(reg_alpha=0.1, reg_lambda=1.0, objective='reg:squarederror')
    model_name = "XGBRegressor (Log Transformation)"

    # ========================================
    # Step 3: Fit the selected model
    print(f"Using model: {model_name}")
    model.fit(X_train, y_train)
    print("Model fitting complete")

    # Step 4: Make predictions on the test set
    test_predictions = model.predict(X_test)

    # If using log transformation for the target, back-transform the predictions
    test_predictions = np.expm1(test_predictions)  # Use this only if log transformation was applied

    # Step 5: Evaluate the model on the training data
    train_predictions = model.predict(X_train)
    # Ensure predictions are non-negative
    test_predictions = np.clip(test_predictions, 0, None)  # Clip to ensure no negative values

    mse = mean_squared_error(y_train, train_predictions)
    r2 = model.score(X_train, y_train)  # Use the model's score method for R²

    print(f"Training Mean Squared Error: {mse}")
    print(f"Training R² Score: {r2}")

    # Step 6: Save predictions to CSV
    # Uncomment the following line to clip negative values to zero
    # test_predictions = np.clip(test_predictions, 0, None)

    test_df['passengers_up_predictions'] = test_predictions
    test_df[['trip_id_unique_station', 'passengers_up_predictions']].to_csv(
        "C:/Users/PC/Documents/Year3SemesterB/67577IML/Hackathon/data/HU.BER/eval_passengers_up.csv",
        index=False
    )
    # Call the plotting functions from the plotting module
    plotting.plot_residuals(y_train, train_predictions)
    plotting.plot_learning_curve(model, X_train, y_train)

    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    print("cross val score: ", scores)
