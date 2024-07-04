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
    rush_hours_list = []
    rush_hours_dict = {}
    for hour in range(24):
        max_passengers = 0
        X_hour = X[X['travel_hour'] == hour]
        for row in X_hour.iterrows():
            if max_passengers == 0 or row[1]['passengers_continue'] > max_passengers:
                max_passengers = row[1]['passengers_continue']
        rush_hours_list.append(max_passengers)
        if max_passengers > 60:
            rush_hours_dict[hour] = max_passengers
    return rush_hours_list, list(rush_hours_dict.keys())


def is_line_in_rush_hour(X):
    rush_hours_per_cluster = {}
    X['arrival_time'] = pd.to_datetime(X['arrival_time'])

    # Extract hour from 'arrival_time'
    X['travel_hour'] = X['arrival_time'].dt.hour

    clusters = X['cluster'].drop_duplicates().values.tolist()
    for cluster in clusters:
        rush_hours_per_cluster[cluster] = identify_rush_hour(X[X['cluster'] == cluster])[1]

    rush_lines_per_cluster = {}
    for cluster in rush_hours_per_cluster.keys():
        filtered_lines = set()
        hour_ranges = rush_hours_per_cluster[cluster]
        filtered = X[(X['cluster'] == cluster)]
        for row in filtered.iterrows():
            if row[1]['travel_hour'] in hour_ranges:
                filtered_lines.add(row[1]['line_id'])
        rush_lines_per_cluster[cluster] = filtered_lines

    # heatmap_data = pd.DataFrame(columns=range(24), index=list(rush_lines_per_cluster.keys()))
    # for cluster, lines in rush_lines_per_cluster.items():
    #     for hour in range(24):
    #         if any(row['travel_hour'] == hour for row in X[X['line_id'].isin(lines)].itertuples()):
    #             heatmap_data.loc[cluster, hour] = 1
    #         else:
    #             heatmap_data.loc[cluster, hour] = 0
    #
    # # Plotting
    # plt.figure(figsize=(12, 8))
    # sns.heatmap(heatmap_data, cmap='Blues', annot=True, fmt='.0f', cbar=True)
    # plt.xlabel('Hour of Day')
    # plt.ylabel('Cluster')
    # plt.title('Presence of Rush Lines by Hour and Cluster')
    # plt.show()

def rush_hour_graph(X: pd.DataFrame, output_path: str = ".") -> typing.NoReturn:
    # Convert 'arrival_time' to datetime format
    rush_hours_per_cluster = {'Hour': list((range(24)))}
    X['arrival_time'] = pd.to_datetime(X['arrival_time'])

    # Extract hour from 'arrival_time'
    X['travel_hour'] = X['arrival_time'].dt.hour

    clusters = X['cluster'].drop_duplicates().values.tolist()
    for cluster in clusters:
        rush_hours_per_cluster[cluster] = identify_rush_hour(X[X['cluster'] == cluster])[0]
    sns.set_style('ticks')
    plt.rcParams.update({'font.size': 8})
    converted_dict = pd.DataFrame(rush_hours_per_cluster)
    rush_hours_per_cluster.pop('Hour')
    palette = sns.color_palette("deep", n_colors=11)
    color_ind = 0

    plt.figure(figsize=(20, 12))
    for cluster in rush_hours_per_cluster.keys():
        if not cluster == 'Hour':
            sns.lineplot(x='Hour', y=cluster, data=converted_dict, marker='o', color=palette[color_ind], label=cluster)
            color_ind += 1
    plt.xticks(ticks=converted_dict['Hour'], labels=converted_dict['Hour'], fontsize=17)
    plt.yticks(fontsize=17)

    plt.title(f'Rush Hours pair Clusters', fontsize=28)
    plt.xlabel('Hour', fontsize=17)
    plt.ylabel('People On The Bus', fontsize=17)
    plt.legend(loc="upper left", fontsize=17)
    plt.grid(True)
    sns.despine()
    plt.savefig(f"{output_path}rush_hour.png")


if __name__ == "__main__":
    df = pd.read_csv("train_bus_schedule_filtered.csv")

    df.dropna()

    # X, y = df.drop('passengers_up', axis=1), df.passengers_up
    # X = df[['passengers_continue_menupach', 'mekadem_nipuach_luz', 'passengers_continue', 'longitude', 'latitude',
    # 'station_id', 'station_index', 'direction', 'line_id']]
    # y = df.passengers_up
    # # random_seed = np.random(0)
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    # feature_evaluation(X_train, y_train)

    # rush_hour_graph(df)
    is_line_in_rush_hour(df)

    # df = df[['trip_id_unique', 'passengers_up', 'passengers_continue', 'cluster', 'station_id']]
    #
    # station_traffic(df, 'A')\

