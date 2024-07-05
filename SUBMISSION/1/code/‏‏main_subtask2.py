
from argparse import ArgumentParser
import logging
import csv
import math
import  first_baseline
from joblib import load
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
import preprocessing
import pandas as pd
from datetime import datetime
from joblib import dump
from geopy.distance import geodesic
from sklearn.linear_model import Ridge, Lasso

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
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

    df['distance_between_stations'] = 0
    df['duration_between_stations'] = 0
    df = df.loc[df['arrival_is_estimated'] == False]

    for i in tqdm(range(len(df) - 1), total=len(df) - 1):
        row1 = df.iloc[i]
        row2 = df.iloc[i + 1]

        if row2["station_index"] == 1:
            continue

        # passenger_cont_is_int_pos
        if row2["passengers_continue"] <= 0:
            df.iloc[i + 1, df.columns.get_loc("passengers_continue")] = 0

        df.iloc[i + 1, df.columns.get_loc('duration_between_stations')] = preprocessing.time_difference(row1["arrival_time"],
                                                                                      row2["arrival_time"])

        # Example usage
        coord1 = (row1["latitude"], row1["longitude"])  # Warsaw, Poland
        coord2 = (row2["latitude"], row2["longitude"])

        df.iloc[i + 1, df.columns.get_loc('distance_between_stations')] = geodesic(coord1, coord2).meters
        scaler = StandardScaler()
        df['duration_between_stations'] = scaler.fit_transform(df[['duration_between_stations']])
        return df

def get_df_for_test(df):
    df['distance_between_stations'] = 0
    # df['duration_between_stations'] = 0
    df = df.loc[df['arrival_is_estimated'] == False]

    for i in tqdm(range(len(df) - 1), total=len(df) - 1):
        row1 = df.iloc[i]
        row2 = df.iloc[i + 1]

        if row2["station_index"] == 1:
            continue

        # passenger_cont_is_int_pos
        if row2["passengers_continue"] <= 0:
            df.iloc[i + 1, df.columns.get_loc("passengers_continue")] = 0

        # df.iloc[i + 1, df.columns.get_loc('duration_between_stations')] = preprocessing.time_difference(row1["arrival_time"],
        #                                                                               row2["arrival_time"])

        # Example usage
        coord1 = (row1["latitude"], row1["longitude"])  # Warsaw, Poland
        coord2 = (row2["latitude"], row2["longitude"])

        df.iloc[i + 1, df.columns.get_loc('distance_between_stations')] = geodesic(coord1, coord2).meters
        # scaler = StandardScaler()
        # df['duration_between_stations'] = scaler.fit_transform(df[['duration_between_stations']])
        return df[['distance_between_stations', "passengers_up", 'trip_id_unique', "station_index"]]


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
    X = df[['distance_between_stations', 'passengers_up', 'trip_id_unique']]
    y = df[['duration_between_stations']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    X_train_model = X_train[['distance_between_stations', 'passengers_up']]
    X_test_model = X_test[['distance_between_stations', 'passengers_up']]

    # Create and fit the model
    model = LinearRegression()
    model.fit(X_train_model, y_train)

    # X_train['passengers_up'] = y_train["passengers_up"]
    # X_y_train_sorted = X_train.sort_values(by='station_id')

    # X_y_train_grouped = X_train.groupby('trip_id_unique')


    # 4. load the test set (args.test_set)
    df_test = pd.read_csv(args.test_set, encoding="ISO-8859-8")

    # 5. preprocess the test set
    logging.info("preprocessing test...")
    X_test = get_df_for_test(df_test)
    # X_test['passengers_up'] = y_test["passengers_up"]
    # X_y_test_sorted = X_test.sort_values(by='station_id')

    # y_train['passengers_up'] = X_train['passengers_up']
    # X_train = X_train.drop('passengers_up', axis=1)
    #
    X_y_test_grouped = X_test.groupby('trip_id_unique')

    # df_gold_standard = pd.DataFrame({
    #     'trip_id_unique_station': X_y_test_sorted["trip_id_unique_station"],
    #     'passengers_up': X_y_test_sorted["passengers_up"]
    # })

    # 6. predict the test set using the trained model
    logging.info("predicting...")

    df_predictions = pd.DataFrame(columns=['trip_id_unique', 'trip_duration_in_minutes'])
    for key, group in tqdm(X_y_test_grouped):

        X_test_model = group[['distance_between_stations', 'passengers_up']]
        # y_test_model = group['passengers_up']
        y_duration_predict = model.predict(X_test_model)
        total_ride = np.sum(y_duration_predict)

        # Create a DataFrame with the predictions
        predictions_df = pd.DataFrame({
            'trip_id_unique': group['trip_id_unique'],
            'trip_duration_in_minutes': total_ride
        })

        # Concatenate the predictions with the main DataFrame
        df_predictions = pd.concat([df_predictions, predictions_df], ignore_index=True)
        df_predictions = df_predictions.drop_duplicates(subset=['trip_id_unique'])

    # 7. save the predictions to args.out
    logging.info("predictions saved to {}".format(args.out))
    df_predictions.to_csv(args.out, index=False)

