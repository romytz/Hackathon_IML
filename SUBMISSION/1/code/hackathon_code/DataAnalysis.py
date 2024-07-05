import typing
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> typing.NoReturn:
    """
        Create scatter plot between each feature and the response.
            - Plot title specifies feature name
            - Plot title specifies Pearson Correlation between feature and response
            - Plot saved under given folder with file name including feature name
        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Design matrix of regression problem

        y : array-like of shape (n_samples, )
            Response vector to evaluate against

        output_path: str (default ".")
            Path to folder in which plots are saved
    """
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
    """
        Calculates and visualizes passenger traffic at each station for bus lines within a specified cluster.

        This function filters the data to a specific cluster, computes the traffic for each station along
        the bus lines, and generates bar charts that show the passenger traffic at each station. The charts
        are saved as PNG files in the specified output path.

        Parameters:
        ----------
        X : pd.DataFrame
            The input DataFrame containing bus trip data. It must include the following columns:
            - 'cluster': Identifier for the cluster.
            - 'line_id': Identifier for the bus line.
            - 'station_id': Identifier for the station.
            - 'passengers_up': Number of passengers getting on the bus at the station.
            - 'passengers_continue': Number of passengers continuing on the bus after the station.

        cluster : int or str
            The specific cluster to analyze. The function will only consider data for this cluster.

        output_path : str, optional
            The directory path where the generated bar charts will be saved. Defaults to the current directory (".").

        Returns:
        -------
        None
            The function does not return any values. It saves bar charts as PNG files in the specified output path.
    """
    lines_stations = {}
    # take all rows that belong to this cluster
    X = X[X['cluster'] == cluster]
    lines_id_list = X.loc[X['cluster'] == cluster, 'line_id'].drop_duplicates().values
    for line_id in lines_id_list:
        stations = X.loc[X['cluster'] == cluster, 'station_id'].drop_duplicates().values
        line_data = X[X['line_id'] == line_id]  # Start from the second row onwards
        stations_traffic = {}
        prev_station_id = 0
        for station_id in stations:
            if prev_station_id is None:
                # Initialize first station's traffic count with the passengers up
                initial_data = line_data[line_data['station_id'] == station_id]
                if not initial_data.empty:
                    stations_traffic[station_id] = initial_data['passengers_up'].sum()
                else:
                    stations_traffic[station_id] = 0
                prev_station_id = station_id
                continue

            # Previous station's data
            prev_data = line_data[line_data['station_id'] == prev_station_id]
            if not prev_data.empty:
                prev_passengers_continue = prev_data['passengers_continue'].sum()
            else:
                prev_passengers_continue = 0

            # Current station's data
            curr_data = line_data[line_data['station_id'] == station_id]
            if not curr_data.empty:
                curr_passengers_up = curr_data['passengers_up'].sum()
                curr_passengers_continue = curr_data['passengers_continue'].sum()
            else:
                curr_passengers_up = 0
                curr_passengers_continue = 0
            # Calculate traffic for current station
            stations_traffic[station_id] = prev_passengers_continue + curr_passengers_up - curr_passengers_continue
            # Update previous station ID
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
            plt.title(f'Traffic at Stations for Line {line_id} in cluster {cluster}')
            plt.savefig(f"{output_path}/traffic_{line_id}.png")  # Save the plot to a file


def identify_rush_hour(X):
    """
        Identifies rush hours based on passenger counts for each hour in a given DataFrame.

        This function analyzes passenger data by hour to determine the hours with the highest
        number of continuing passengers, which are considered rush hours. The rush hours are
        defined as those hours where the number of continuing passengers exceeds a specific
        threshold (60 in this case).

        Parameters:
        ----------
        X : pd.DataFrame
            The input DataFrame containing bus trip data. The DataFrame must include the following columns:
            - 'travel_hour': An integer representing the hour of the day when the trip took place (0-23).
            - 'passengers_continue': An integer representing the number of passengers continuing on the bus after a given station.

        Returns:
        -------
        tuple:
            A tuple containing two elements:
            - A list of integers (`rush_hours_list`), where each entry represents the maximum number of
              continuing passengers for each hour of the day (0 to 23).
            - A list of integers (`rush_hours_dict.keys()`), representing the hours considered as rush
              hours, where the number of continuing passengers exceeds the threshold of 60.
    """
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
    """
        Analyzes bus lines to identify rush hour lines per cluster and visualizes the results.

        This function calculates rush hour lines for each cluster based on the number of continuing passengers
        during specific hours of the day. It plots a bar chart showing the number of rush hour lines per cluster
        and saves the plot as 'Rush_Hours_per_Cluster.png'.

        Parameters:
        ----------
        X : pd.DataFrame
            The input DataFrame containing bus trip data. It must include the following columns:
            - 'arrival_time': The timestamp of when the bus arrived at a station.
            - 'cluster': Identifier for the cluster to which the bus trip belongs.
            - 'line_id': Identifier for the bus line.
            - 'travel_hour': Derived from 'arrival_time', represents the hour of the day (0-23).

        Returns:
        -------
        None
            This function does not return any value. It generates and saves a bar chart showing the number of
            rush hour lines per cluster.
    """
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

    clusters = sorted(rush_lines_per_cluster.keys())
    rush_counts = [len(rush_lines_per_cluster[cluster]) for cluster in clusters]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(clusters, rush_counts, color='skyblue')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Rush Hour Lines')
    plt.title('Number of Rush Hour Lines per Cluster')
    plt.xticks(rotation=55)
    plt.tight_layout()
    plt.savefig("Rush_Hours_per_Cluster.png")


def rush_hour_graph(X: pd.DataFrame, output_path: str = ".") -> typing.NoReturn:
    """
        Generates a line plot to visualize rush hour patterns for each cluster based on bus trip data.

        This function analyzes bus trip data to identify rush hours for each cluster. It then plots a line
        graph showing the number of people on the bus during rush hours for each cluster. The plot is saved
        as 'rush_hour.png' in the specified output path.

        Parameters:
        ----------
        X : pd.DataFrame
            The input DataFrame containing bus trip data. It must include the following columns:
            - 'arrival_time': The timestamp of when the bus arrived at a station.
            - 'cluster': Identifier for the cluster to which the bus trip belongs.
            - 'passengers_continue': Number of passengers continuing on the bus after a station.

        output_path : str, optional
            The directory where the output plot will be saved. Default is the current directory ('.').

        Returns:
        -------
        None
            This function does not return any value. It generates and saves a line plot showing rush hour
            patterns for each cluster.

        Notes:
        -----
        - 'arrival_time' is converted to datetime format to extract the hour ('travel_hour').
        - Rush hours are identified using the 'identify_rush_hour' function, which determines hours with
          a high number of continuing passengers.
        - Each cluster's rush hour patterns are visualized using seaborn's lineplot.
        - The plot is customized with appropriate labels, title, legend, and grid.
    """
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

    # Tasks - Part 3.1, 3.2
    X, y = df.drop('passengers_up', axis=1), df.passengers_up
    X = df[['trip_id', 'line_id', 'direction', 'station_index', 'station_id', 'latitude', 'longitude',
            'passengers_up', 'passengers_continue', 'mekadem_nipuach_luz', 'passengers_continue_menupach',
            'time_in_station (sec)']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    feature_evaluation(X_train, y_train)

    # Tasks - Part 3.3
    rush_hour_graph(df)
    is_line_in_rush_hour(df)

    station_traffic(df, 'Mizrahi-Ramat gan')
