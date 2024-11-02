from typing import Tuple
import pandas as pd
import numpy as np

# Load the original dataset for calculating mean values later
original_data = pd.read_csv("data/train_bus_schedule_duration.csv", encoding="ISO-8859-8")

# Initialize global variables for mean longitude and latitude
longitude_mean = 0
latitude_mean = 0

def preprocess_train(X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
       Preprocesses the training data by filling missing values, removing NaNs, and calculating mean locations.

       Args:
           X (pd.DataFrame): The training data DataFrame.

       Returns:
           Tuple[pd.DataFrame, pd.Series]: The preprocessed features and the target variable.
       """
    # Fill missing 'door_closing_time' values with 'arrival_time'
    X["door_closing_time"] = X["door_closing_time"].fillna(X["arrival_time"])
    # Drop rows with any remaining NaN values
    X = X.dropna()

    # Calculate and set global mean for longitude and latitude for further feature engineering
    global longitude_mean, latitude_mean
    longitude_mean = X["longitude"].mean()
    latitude_mean = X["latitude"].mean()

    # Call mutual preprocessing function for further processing
    X_preprocessed, y_preprocessed, trip_id_unique_station = mutual_preprocess(X)
    return X_preprocessed, y_preprocessed


def preprocess_test(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
        Preprocesses the test data by filling missing values and calling mutual preprocessing function.

        Args:
            X (pd.DataFrame): The test data DataFrame.

        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.Series]: Preprocessed features, an empty target (None), and unique trip IDs.
        """
    # Fill missing 'door_closing_time' values with 'arrival_time'
    X["door_closing_time"] = X["door_closing_time"].fillna(X["arrival_time"])
    # Call mutual preprocessing function for further processing
    return mutual_preprocess(X)


def mutual_preprocess(X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
        Common preprocessing steps for both training and test sets, including feature engineering and filtering.

        Args:
            X (pd.DataFrame): The input data (either training or test DataFrame).

        Returns:
            Tuple[pd.DataFrame, pd.Series, pd.Series]: Processed features, target variable (if available), and trip IDs.
        """
    # Filter rows with valid 'passengers_up' and 'duration' values
    X = X[(X["passengers_up"] >= 0) & (~X["passengers_up"].isnull())]
    if "duration" in X.columns:
        X = X.loc[(X["duration"] > 0) & (X["duration"] <= 180)]

    # Aggregate data by 'trip_id_unique' to calculate feature values
    grouped_by_trip_unique = X.groupby("trip_id_unique")
    X['num_stations'] = grouped_by_trip_unique['station_index'].transform(lambda x: x.max())
    X["total_passengers_up"] = grouped_by_trip_unique['passengers_up'].transform(lambda x: x.sum())

    # Calculate distance from mean and distance from the center of the earth
    X["distance_from_mean"] = ((X["longitude"] - longitude_mean) ** 2 + (X["latitude"] - latitude_mean) ** 2) ** 0.5
    X["dist_from_center_of_the_earth"] = ((X["longitude"] - 0) ** 2 + (X["latitude"] - 0) ** 2) ** 0.5

    # Log transformation for skewed features
    X["total_passengers_up"] = np.log1p(X["total_passengers_up"])
    X["num_stations"] = np.log1p(X["num_stations"])

    # Interaction feature: cumulative effect of passengers and station count
    X["interaction_passengers_stations"] = X["total_passengers_up"] * X["num_stations"]

    # Time-based features extraction from 'arrival_time' if it exists
    if 'arrival_time' in X.columns:
        X['arrival_time'] = pd.to_datetime(X['arrival_time'])
        X['hour'] = X['arrival_time'].dt.hour
        X['day_of_week'] = X['arrival_time'].dt.dayofweek

    # Create parts of the day (morning, afternoon, evening, night)
    def part_of_day(hour):
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 21:
            return 'evening'
        else:
            return 'night'

    if 'hour' in X.columns:
        X['part_of_day'] = X['hour'].apply(part_of_day)

        # One-hot encode the part of the day
        X = pd.get_dummies(X, columns=['part_of_day'], drop_first=True)

    # Drop duplicate rows based on 'trip_id_unique' to keep unique trips
    X = X.drop_duplicates(subset=["trip_id_unique"])

    # Define categorical features to expand and keep
    categorical_features_to_keep = ["line_id"]
    for feature in categorical_features_to_keep:
        X = create_categorical_columns(feature, X)
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    X['cluster'] = le.fit_transform(X['cluster'])
    X['part'] = le.fit_transform(X['part'])

    # Define features to retain after processing
    features_to_keep = ["part", "cluster", "direction", "duration", "trip_id_unique", "num_stations", "total_passengers_up", "interaction_passengers_stations", "hour", "day_of_week"]
    features_to_keep += [fname + "_" + str(i) for fname in categorical_features_to_keep for i in range(1, len(original_data[fname].unique()))]
    features_to_keep += [col for col in X.columns if 'part_of_day_' in col]  # Keep part_of_day one-hot encoded columns

    # Drop columns that are not in 'features_to_keep'
    feature_names = [feature for feature in X.columns if feature not in features_to_keep]
    X = X.drop(columns=feature_names)

    # Separate features and target variable (if it exists in the data)
    y_preprocessed = X["duration"] if "duration" in X.columns else None
    X_preprocessed = X.drop(columns=["duration", "trip_id_unique"]) if "duration" in X.columns else X.drop(columns=["trip_id_unique"])
    trip_id_unique_station = X["trip_id_unique"]

    return X_preprocessed, y_preprocessed, trip_id_unique_station


def create_categorical_columns(col_name: str, X: pd.DataFrame) -> pd.DataFrame:
    """
        Convert a categorical column into multiple binary columns using one-hot encoding.

        Args:
            col_name (str): The name of the categorical column to convert.
            X (pd.DataFrame): The DataFrame in which the column is located.

        Returns:
            pd.DataFrame: The modified DataFrame with one-hot encoded columns.
        """
    # Create binary columns for each unique value in the categorical column
    for i, val in enumerate(sorted(original_data[col_name].unique())):
        new_cols = pd.DataFrame()
        new_cols[col_name + "_" + str(i)] = (X[col_name] == val).astype(int)
        X = pd.concat([X, new_cols], axis=1)
    # Drop the original categorical column
    X = X.drop(col_name, axis=1)
    return X
