import csv
import math
import  first_baseline
from joblib import load

import numpy as np
from tqdm import tqdm
import preprocessing
import pandas as pd
from datetime import datetime
pd.options.mode.copy_on_write = True
from joblib import dump

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def eval_boardings(predictions: pd.DataFrame, ground_truth: pd.DataFrame):
    combined = pd.merge(predictions, ground_truth, on='trip_id_unique_station')
    mse_board = mean_squared_error(combined["passengers_up_x"], combined["passengers_up_y"])
    return mse_board

def get_df_for_test(df):
    df['time_in_station (sec)'] = 0

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):

        if pd.isna(row["door_closing_time"]):
            df.iloc[index, df.columns.get_loc("door_closing_time")] = row["arrival_time"]
            row["door_closing_time"] = row["arrival_time"]

        if not preprocessing.is_time_after(row["door_closing_time"], row["arrival_time"]):
            df.iloc[index, df.columns.get_loc("door_closing_time")], df.iloc[index, df.columns.get_loc("arrival_time")] = row["arrival_time"], row[
                "door_closing_time"]

        # passenger_cont_is_int_pos
        if row["passengers_continue"] <= 0:
            df.iloc[index, df.columns.get_loc("passengers_continue")] = 0

        df.iloc[index, df.columns.get_loc('time_in_station (sec)')] = preprocessing.time_difference(row["arrival_time"], row["door_closing_time"])

        return df[['time_in_station (sec)', "passengers_continue", "door_closing_time", 'trip_id_unique_station', "station_id"]]


if __name__ == '__main__':

    # baseline_model = first_baseline.get_baseline_model()
    # dump(baseline_model, 'linear_regression_model.joblib')
    baseline_model = load('linear_regression_model.joblib')

    # Replace 'path_to_file.csv' with the actual path to your CSV file
    df = pd.read_csv(r'C:\Elchanan\masters\second_year\Semester_B\IML\hackathon\train_bus_schedule_filtered.csv',
                     encoding="utf-8")

    X = df[['time_in_station (sec)', 'passengers_continue', 'trip_id_unique_station', "station_id"]]
    y = df[['passengers_up']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train['passengers_up'] = y_train["passengers_up"]
    X_y_train_sorted = X_train.sort_values(by='station_id')

    # y_train['passengers_up'] = X_train['passengers_up']
    # X_train = X_train.drop('passengers_up', axis=1)
    #
    X_y_train_grouped = X_y_train_sorted.groupby('station_id')
    model_per_stations_dict = {}
    i = 0

    # grouped_list = list(X_y_train_sorted)

    # Wrap the list with tqdm for a progress bar
    for key, group in tqdm(X_y_train_grouped):

        X_train_model = group[['time_in_station (sec)', 'passengers_continue']]
        y_train_model = group['passengers_up']

        # Create and fit the model
        model = LinearRegression()
        model.fit(X_train_model, y_train_model)
        model_per_stations_dict[key] = model
        # i += 1
        #
        # if i == 5:
        #     break


    ########## PREDICTIONS ##########################
    df_test = pd.read_csv(r'C:\Elchanan\masters\second_year\Semester_B\IML\hackathon\HU.BER\X_passengers_up.csv',
                     encoding="ISO-8859-8")
    X_test = get_df_for_test(df_test)
    # X_test['passengers_up'] = y_test["passengers_up"]
    X_y_test_sorted = X_test.sort_values(by='station_id')


    # y_train['passengers_up'] = X_train['passengers_up']
    # X_train = X_train.drop('passengers_up', axis=1)
    #
    X_y_test_grouped = X_y_test_sorted.groupby('station_id')

    # df_gold_standard = pd.DataFrame({
    #     'trip_id_unique_station': X_y_test_sorted["trip_id_unique_station"],
    #     'passengers_up': X_y_test_sorted["passengers_up"]
    # })

    df_predictions = pd.DataFrame(columns=['trip_id_unique_station', 'passengers_up'])
    for key, group in tqdm(X_y_test_grouped):

        X_test_model = group[['time_in_station (sec)', 'passengers_continue']]
        # y_test_model = group['passengers_up']

        if key not in model_per_stations_dict.keys():
            model = baseline_model
            y_station_predict = model.predict(X_test_model)
            y_station_predict = y_station_predict.flatten().astype(float)
        else:
            # Create and fit the model
            model = model_per_stations_dict[key]
            y_station_predict = model.predict(X_test_model)

        # Create a DataFrame with the predictions
        predictions_df = pd.DataFrame({
            'trip_id_unique_station': group['trip_id_unique_station'],
            'passengers_up': y_station_predict
        })

        # Concatenate the predictions with the main DataFrame
        df_predictions = pd.concat([df_predictions, predictions_df], ignore_index=True)

    df_predictions.to_csv('passengers_up_predictions.csv', index=False)

    # Evaluate the model
    # mse = mean_squared_error(y_test, y_pred)
    # mse_boarding = eval_boardings(df_predictions, df_gold_standard)
    # print(f"MSE for boardings: {mse_boarding}")

    # print(f"Mean Squared Error: {mse}")



    # for index, row in tqdm(df.iterrows(), total=df.shape[0]):





