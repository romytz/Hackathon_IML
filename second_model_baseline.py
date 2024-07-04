import csv
import math
import  first_baseline
from joblib import load

import numpy as np
from tqdm import tqdm
import preprocessing
import pandas as pd
from datetime import datetime
from joblib import dump
from geopy.distance import geodesic
from sklearn.linear_model import Ridge

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
def eval_duration(predictions: pd.DataFrame, ground_truth: pd.DataFrame):
    combined = pd.merge(predictions, ground_truth, on='trip_id_unique')
    mse_loss = mean_squared_error(combined["trip_duration_in_minutes_x"], combined["trip_duration_in_minutes_y"])
    return mse_loss

pd.options.mode.copy_on_write = True

# df = pd.read_csv(r'C:\Elchanan\masters\second_year\Semester_B\IML\hackathon\train_bus_schedule_filtered.csv',
#                  encoding="utf-8")
# df['distance_between_stations'] = 0
# df['duration_between_stations'] = 0
#
# for i in tqdm(range(len(df) - 1), total=len(df) - 1):
#     row1 = df.iloc[i]
#     row2 = df.iloc[i + 1]
#
#     if row2["station_index"] == 1:
#         continue
#
#     # passenger_cont_is_int_pos
#     if row2["passengers_continue"] <= 0:
#         df.iloc[i + 1, df.columns.get_loc("passengers_continue")] = 0
#
#     df.iloc[i + 1, df.columns.get_loc('duration_between_stations')] = preprocessing.time_difference(row1["arrival_time"],
#                                                                                   row2["arrival_time"])
#
#     # Example usage
#     coord1 = (row1["latitude"], row1["longitude"])  # Warsaw, Poland
#     coord2 = (row2["latitude"], row2["longitude"])
#
#     df.iloc[i + 1, df.columns.get_loc('distance_between_stations')] = geodesic(coord1, coord2).meters
#
# with open('df_of_question_2.pkl', 'wb') as file:
#     pickle.dump(df, file)
with open('df_of_question_2.pkl', 'rb') as file:
    df = pickle.load(file)

X = df[['distance_between_stations', 'passengers_up', 'trip_id_unique']]
y = df[['duration_between_stations']]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train_model = X_train[['distance_between_stations', 'passengers_up']]
X_test_model = X_test[['distance_between_stations', 'passengers_up']]

# Create and fit the model
model = Ridge(alpha=0.0)
model.fit(X_train_model, y_train)

# Make predictions
y_pred = model.predict(X_test_model)

df_predictions = pd.DataFrame({
    'trip_id_unique': X_test["trip_id_unique"]})

df_predictions['trip_duration_in_minutes'] = y_pred

df_gold_standard = pd.DataFrame({
    'trip_id_unique': X_test["trip_id_unique"],
    'trip_duration_in_minutes': y_test["duration_between_stations"]
})

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mse_boarding = eval_duration(df_predictions, df_gold_standard)
print(f"MSE for boardings: {mse_boarding}")

print(f"Mean Squared Error: {mse}")

print("that mean error of ", math.sqrt(mse_boarding), " seconds")


