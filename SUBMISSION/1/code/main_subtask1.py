import logging
from joblib import load
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
from argparse import ArgumentParser

def is_time_after(time1, time2, time_format='%H:%M:%S'):
    try:
        t1 = datetime.strptime(time1, time_format)
        t2 = datetime.strptime(time2, time_format)
        return t1 >= t2
    except ValueError:
        return False  # In case of invalid time format

def time_difference(time1, time2, time_format='%H:%M:%S'):
    t1 = datetime.strptime(time1, time_format)
    t2 = datetime.strptime(time2, time_format)
    delta = t2 - t1
    return int(delta.total_seconds())

def preprocess_data(df):
    df = df.drop("station_name", axis=1)
    df['time_in_station (sec)'] = 0

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):

        # df.iloc[index, df.columns.get_loc("part")] = part_dict[row["part"]]
        # df.iloc[index, df.columns.get_loc("cluster")] = cluster_dict[row["cluster"]]

        if pd.isna(row["door_closing_time"]):
            df.iloc[index, df.columns.get_loc("door_closing_time")] = row["arrival_time"]
            row["door_closing_time"] = row["arrival_time"]

        if not is_time_after(row["door_closing_time"], row["arrival_time"]):
            df.iloc[index, df.columns.get_loc("door_closing_time")], df.iloc[
                index, df.columns.get_loc("arrival_time")] = row["arrival_time"], row[
                "door_closing_time"]

        if pd.isna(row["door_closing_time"]):
            df.iloc[index, df.columns.get_loc("door_closing_time")] = row["arrival_time"]
            row["door_closing_time"] = row["arrival_time"]

        # close_after_arr
        if not is_time_after(row["door_closing_time"], row["arrival_time"]):
            df.iloc[index, df.columns.get_loc("door_closing_time")], df.iloc[
                index, df.columns.get_loc("arrival_time")] = row["arrival_time"], row[
                "door_closing_time"]

        # passenger_cont_is_int_pos
        if row["passengers_continue"] <= 0:
            df.iloc[index, df.columns.get_loc("passengers_continue")] = 0

        df.iloc[index, df.columns.get_loc('time_in_station (sec)')] = time_difference(row["arrival_time"],
                                                                                      row["door_closing_time"])

        return df

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


"""
usage:
    python code/main.py --training_set PATH --test_set PATH --out PATH

for example:
    python code/main.py --training_set /cs/usr/gililior/training.csv --test_set /cs/usr/gililior/test.csv --out predictions/trip_duration_predictions.csv 

"""

# implement here your load,preprocess,train,predict,save functions (or any other design you choose)


if __name__ == '__main__':
    print('Running')
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True,
                        help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True,
                        help="path to the test set")
    parser.add_argument('--out', type=str, required=True,
                        help="path of the output file as required in the task description")
    args = parser.parse_args()

    # 1. load the training set (args.training_set)
    df = pd.read_csv(args.training_set, encoding="ISO-8859-8")

    # 2. preprocess the training set
    logging.info("preprocessing train...")
    df = preprocess_data(df)

    # 3. train a model
    logging.info("training...")
    # this model was trained in the same wat+y, but without splitting into stations (less expressive)
    baseline_model = load('linear_regression_model.joblib')

    X_train = df[['time_in_station (sec)', 'passengers_continue', 'trip_id_unique_station', "station_id"]]
    y_train = df[['passengers_up']]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0, random_state=42)

    X_train['passengers_up'] = y_train["passengers_up"]
    X_y_train_sorted = X_train.sort_values(by='station_id')

    X_y_train_grouped = X_y_train_sorted.groupby('station_id')
    model_per_stations_dict = {}

    # Wrap the list with tqdm for a progress bar
    for key, group in tqdm(X_y_train_grouped):
        X_train_model = group[['time_in_station (sec)', 'passengers_continue']]
        y_train_model = group['passengers_up']

        # Create and fit the model
        model = LinearRegression()
        model.fit(X_train_model, y_train_model)
        model_per_stations_dict[key] = model


    # 4. load the test set (args.test_set)
    df_test = pd.read_csv(args.test_set, encoding="ISO-8859-8")

    # 5. preprocess the test set
    logging.info("preprocessing test...")
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

    # 6. predict the test set using the trained model
    logging.info("predicting...")

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

    # 7. save the predictions to args.out
    logging.info("predictions saved to {}".format(args.out))
    df_predictions.to_csv(args.out, index=False)

