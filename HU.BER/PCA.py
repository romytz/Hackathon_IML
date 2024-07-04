import typing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pandas
import pandas as pd
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> typing.NoReturn:
    for feature in X.columns:
        # compute the Pearson correlation coefficient
        correlation = X[feature].corr(y)

        # create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(X[feature], y, alpha=0.5)
        plt.title(f'{feature} - Pearson Correlation: {correlation:.2f}')
        plt.xlabel(feature)
        plt.ylabel('Passengers up')
        plt.savefig(f'Pearson_Corr{output_path}{feature}.png')
        plt.close()


def station_traffic(X: pd.DataFrame, cluster, output_path: str = ".") -> typing.NoReturn:
    lines_stations = {}
    # take all rows that belong to this cluster
    X = X[X['cluster'] == cluster]
    lines_id_list = X.loc[X['cluster'] == cluster, 'trip_id_unique'].drop_duplicates().values
    stations = X.loc[X['cluster'] == cluster, 'station_id'].drop_duplicates().values
    for line_id in lines_id_list:
        line_data = X[X['trip_id_unique'] == line_id]  # Start from the second row onwards
        stations_traffic = {}
        prev_station_id = 0
        for station_id in stations:
            if not stations_traffic:
                # number of people that visit the first station
                prev_station_id = station_id
                stations_traffic[station_id] = 0
                continue
            x = line_data[line_data['station_id'] == prev_station_id]
            prev_passengers_continue = x['passengers_continue'].values[0]
            y = line_data[line_data['station_id'] == station_id]
            curr_passengers_up = y['passengers_up'].values[0]
            z = line_data[line_data['station_id'] == station_id]
            curr_passengers_continue = z['passengers_continue'].values[0]
            stations_traffic[station_id] = prev_passengers_continue + curr_passengers_up - curr_passengers_continue
            prev_station_id = station_id
        lines_stations[line_id] = stations_traffic

    for line_id, station_traffic in lines_stations.items():
        filtered_stations_traffic = {str(k): v for k, v in station_traffic.items() if v > 0}

        if filtered_stations_traffic:
            new_stations = list(filtered_stations_traffic.keys())
            new_traffic = list(filtered_stations_traffic.values())
            plt.figure(figsize=(10, 6))
            plt.bar(new_stations, new_traffic, color='lightblue')
            plt.xlabel('Station ID')
            plt.ylabel('Traffic')
            plt.title(f'Traffic at Stations for Line {line_id}')
            plt.savefig(f"{output_path}/traffic_{line_id}.png")  # Save the plot to a file


# Function to determine rush hour period
def identify_rush_hour(X):
    # Define rush hour criteria (adjust as needed)
    rush_hours = {}
    for hour in range(24):
        max_passengers = 0
        X_hour = X[X['travel_hour'] == hour]
        for row in X_hour:
            if max_passengers == 0 or row['passengers_continue'] > max_passengers:
                max_passengers = row['passengers_continue']
        if max_passengers > 20:
            rush_hours[hour] = max_passengers
    return rush_hours.keys()


def is_rush_hour(hour, rush_hours):
    return hour in rush_hours


def rush_hour(X: pd.DataFrame, cluster, output_path: str = ".") -> typing.NoReturn:
    # Convert 'arrival_time' to datetime format
    X['arrival_time'] = pd.to_datetime(X['arrival_time'])

    # Extract hour from 'arrival_time'
    X['travel_hour'] = X['arrival_time'].dt.hour

    rush_hours = identify_rush_hour(X)

    # Apply rush hour determination and group by cluster
    X['is_rush_hour'] = X['travel_hour'].apply(is_rush_hour)
    print(X)



if __name__ == "__main__":
    df = pd.read_csv("train_bus_schedule.csv", encoding="ISO-8859-8")

    df.dropna()

    # X, y = df.drop('passengers_up', axis=1), df.passengers_up
    # X = df[['passengers_continue_menupach', 'mekadem_nipuach_luz', 'passengers_continue', 'longitude', 'latitude',
    # 'station_id', 'station_index', 'direction', 'line_id']]
    # y = df.passengers_up
    # # random_seed = np.random(0)
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    # feature_evaluation(X_train, y_train)


    df = pd.read_csv("train_bus_schedule.csv", encoding="ISO-8859-8")
    df.dropna()
    df.drop('station_name', axis=1)
    df = df[df['cluster'] == 'A']
    rush_hour(df, 'A')
    # df = df[['trip_id_unique', 'passengers_up', 'passengers_continue', 'cluster', 'station_id']]
    #
    # station_traffic(df, 'A')\

