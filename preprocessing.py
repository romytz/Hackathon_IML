# import pandas as pd
#
#
# def preprocess_data(df: pd.DataFrame, is_train=True) -> pd.DataFrame:
#     # Fill missing door_closing_time with arrival_time
#     df['door_closing_time'] = df['door_closing_time'].fillna(df['arrival_time'])
#
#     # Convert arrival_time and door_closing_time to seconds since midnight
#     df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='%H:%M:%S', errors='coerce').dt.time
#     df['door_closing_time'] = pd.to_datetime(df['door_closing_time'], format='%H:%M:%S', errors='coerce').dt.time
#
#     def time_to_seconds(t):
#         if pd.isnull(t):
#             return 0
#         return t.hour * 3600 + t.minute * 60 + t.second
#
#     df['arrival_time'] = df['arrival_time'].apply(time_to_seconds)
#     df['door_closing_time'] = df['door_closing_time'].apply(time_to_seconds)
#
#     # Handle categorical columns (e.g., cluster, trip_id_unique_station)
#     df['cluster'] = df['cluster'].astype('category').cat.codes
#     df['trip_id_unique_station'] = df['trip_id_unique_station'].astype('category').cat.codes
#
#     # Drop unnecessary columns
#     columns_to_drop = ['trip_id', 'trip_id_unique', 'station_name', 'part', 'latitude', 'longitude']
#     df = df.drop(columns=columns_to_drop, axis=1)
#
#     # Handle missing values in numeric columns
#     df = df.fillna(0)
#
#     # Return the preprocessed dataframe
#     return df


import pandas as pd

def preprocess_data(df: pd.DataFrame, is_train=True) -> pd.DataFrame:
    # Fill missing door_closing_time with arrival_time
    df['door_closing_time'] = df['door_closing_time'].fillna(df['arrival_time'])

    # Convert arrival_time and door_closing_time to seconds since midnight
    df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='%H:%M:%S', errors='coerce').dt.time
    df['door_closing_time'] = pd.to_datetime(df['door_closing_time'], format='%H:%M:%S', errors='coerce').dt.time

    def time_to_seconds(t):
        if pd.isnull(t):
            return 0
        return t.hour * 3600 + t.minute * 60 + t.second

    df['arrival_time'] = df['arrival_time'].apply(time_to_seconds)
    df['door_closing_time'] = df['door_closing_time'].apply(time_to_seconds)

    # Handle categorical columns (e.g., cluster, trip_id_unique_station)
    df['cluster'] = df['cluster'].astype('category').cat.codes
    df['trip_id_unique_station'] = df['trip_id_unique_station'].astype('category').cat.codes

    # Drop unnecessary columns
    columns_to_drop = ['trip_id', 'trip_id_unique', 'station_name', 'part', 'latitude', 'longitude']
    df = df.drop(columns=columns_to_drop, axis=1)

    # Handle missing values in numeric columns
    df = df.fillna(0)

    # Detect and handle outliers using IQR (Interquartile Range)
    def handle_outliers(col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Cap the outliers
        df[col] = df[col].apply(lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))

    # List of numeric columns where we want to handle outliers
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    for col in numeric_columns:
        handle_outliers(col)

    # Return the preprocessed dataframe
    return df
