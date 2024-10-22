from typing import Tuple
import pandas as pd
import datetime

# Load the original dataset
original_data = pd.read_csv(
    r"C:\Users\PC\Documents\Year3SemesterB\67577IML\Hackathon\data\HU.BER\train_bus_schedule.csv", encoding="ISO-8859-8")


def preprocess_train(X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Fill missing door_closing_time with arrival_time
    X["door_closing_time"] = X["door_closing_time"].fillna(X["arrival_time"])

    # Drop rows with any remaining missing values
    X = X.dropna()

    # Remove outliers in passengers_up and passengers_continue
    X = X[X["passengers_up"] < 30]
    X = X[X["passengers_continue"] < 60]
    X = X[X["passengers_continue"] > 0]

    # Preprocess and return the result
    X_preprocessed, y_preprocessed, trip_id_unique_station = mutual_preprocess(X)

    X_preprocessed.to_csv(r'C:\Users\PC\Documents\Year3SemesterB\67577IML\Hackathon\data\HU.BER\preprocess_train.csv', index=False)
    return X_preprocessed, y_preprocessed


def preprocess_test(X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    # Fill missing door_closing_time with arrival_time
    X["door_closing_time"] = X["door_closing_time"].fillna(X["arrival_time"])

    # Preprocess and return the result
    return mutual_preprocess(X)

cluster_dict = {
    "אונו-אלעד": "Ono-Elad", "בת ים-רמת גן": "Bat yam - Ramat gan",
    "דרומי-בת ים": "Dromi-Bat yam", 'דרומי-ראשל"צ-חולון': "Dromi-Rashlatz_Holon",
    'השרון': 'Hasharon', 'חולון עירוני ומטרופוליני+תחרות חולון': "Holon Metropolin",
    'מזרחי-בני ברק': 'Mizrahi-Bnei brak', 'מזרחי-רמת גן': "Mizrahi-Ramat gan",
    'פ"ת-ת"א': "Pat-TLV", 'שרון חולון מרחבי': 'Sharon-Holon-Merhavi', 'תל אביב': "TLV"
}

def mutual_preprocess(X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    # Drop all invalid passengers_up values if present
    if "passengers_up" in X.columns:
        X = X[(X["passengers_up"] >= 0) & (~X["passengers_up"].isnull())]

    # Drop redundant columns
    X = X.drop(["part", "trip_id_unique", "station_name", "latitude", "longitude"], axis=1)

    # Convert arrival_is_estimated to integer type
    X["arrival_is_estimated"] = X["arrival_is_estimated"].astype(int)

    # Convert "arrival_time" and "door_closing_time" to datetime format
    X["arrival_time"] = pd.to_datetime(X["arrival_time"], format='%H:%M:%S').dt.time
    X["door_closing_time"] = pd.to_datetime(X["door_closing_time"], format='%H:%M:%S').dt.time

    # Handle missing times: replace null arrival_time with 00:00:00
    X.loc[X["arrival_time"].isnull(), "arrival_time"] = datetime.time(0, 0, 0)
    X.loc[X["door_closing_time"].isnull(), "door_closing_time"] = X["arrival_time"]

    # Ensure door closing time is after arrival time
    for index, row in X.iterrows():
        X.at[index, 'cluster'] = cluster_dict.get(row['cluster'], row['cluster'])
        if not is_time_after(row["door_closing_time"], row["arrival_time"]):
            # Swap the times if door closing time is before arrival time
            X.at[index, "door_closing_time"], X.at[index, "arrival_time"] = row["arrival_time"], row["door_closing_time"]

    # Calculate time_in_station in seconds and ensure it's non-negative
    X['time_in_station (sec)'] = [
        max(time_difference(row["arrival_time"], row["door_closing_time"]), 0)
        for _, row in X.iterrows()
    ]

    # Alternative feature transformation (if needed)
    X['alternative'] = X['alternative'].apply(lambda x: int(x) if x.isdigit() else 0)

    y_preprocessed = None
    if "passengers_up" in X.columns:
        X_preprocessed = X.drop(columns=["passengers_up", "trip_id_unique_station"])
    else:
        X_preprocessed = X.drop(columns=["trip_id_unique_station"])

    if "passengers_up" in X.columns:
        y_preprocessed = X["passengers_up"]

    trip_id_unique_station = X["trip_id_unique_station"]

    return X_preprocessed, y_preprocessed, trip_id_unique_station


def time_difference(time1, time2, time_format='%H:%M:%S'):
    """
    Calculate the time difference in seconds between time1 and time2.

    Args:
        time1 (str): First time in string format.
        time2 (str): Second time in string format.
        time_format (str): Format of the time string, default is '%H:%M:%S'.

    Returns:
        int: Time difference in seconds, or None if time1 or time2 is invalid.
    """
    try:
        t1 = datetime.datetime.strptime(str(time1), '%H:%M:%S')
        t2 = datetime.datetime.strptime(str(time2), '%H:%M:%S')
        delta = t2 - t1
        return int(delta.total_seconds())
    except (ValueError, TypeError):
        return None  # Handle invalid time format


def is_time_after(time1, time2):
    """
    Check if time1 is later or equal to time2.

    Args:
        time1 (str): First time in string format.
        time2 (str): Second time in string format.

    Returns:
        bool: True if time1 is later or equal to time2, False otherwise.
    """
    try:
        t1 = datetime.datetime.strptime(str(time1), '%H:%M:%S')
        t2 = datetime.datetime.strptime(str(time2), '%H:%M:%S')
        return t1 >= t2
    except (ValueError, TypeError):
        return False




# from typing import Tuple
# import pandas as pd
# # from datetime import datetime, time
# import datetime
#
# original_data = pd.read_csv(
#     "C:\Users\PC\Documents\Year3SemesterB\67577IML\Hackathon\data\HU.BER\train_bus_schedule.csv", encoding="ISO-8859-8")
#
#
# def preprocess_train(X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
#     X["door_closing_time"] = X["door_closing_time"].fillna(X["arrival_time"])
#     X = X.dropna()
#     # outliers
#     X = X[X["passengers_up"] < 30]
#     X = X[X["passengers_continue"] < 60]
#     X = X[X["passengers_continue"] > 0]
#     X_preprocessed, y_preprocessed, trip_id_unique_station = mutual_preprocess(X)
#     # X_preprocessed = X_preprocessed.drop_duplicates()
#     return X_preprocessed, y_preprocessed
#
#
# def preprocess_test(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
#     X["door_closing_time"] = X["door_closing_time"].fillna(X["arrival_time"])
#     return mutual_preprocess(X)
#
#
# def mutual_preprocess(X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
#     # Drop all invalid passengers_up
#     if "passengers_up" in X.columns:
#         X = X[(X["passengers_up"] >= 0) & (~X["passengers_up"].isnull())]
#
#     # Drop redundant columns
#     X = X.drop(["part", "trip_id_unique", "station_name", "latitude", "longitude"], axis=1)
#
#     X["arrival_is_estimated"] = X["arrival_is_estimated"].astype(int)
#
#     # Convert the "arrival_time" column to datetime format
#     X["arrival_time"] = pd.to_datetime(X["arrival_time"], format='%H:%M:%S').dt.time
#     X["door_closing_time"] = pd.to_datetime(X["door_closing_time"], format='%H:%M:%S').dt.time
#
#     X.loc[X["arrival_time"].isnull(), "arrival_time"] = datetime.time(0, 0, 0)
#     X.loc[X["door_closing_time"].isnull(), "door_closing_time"] = X["arrival_time"]
#
#     X['alternative'] = X['alternative'].apply(lambda x: int(x) if x.isdigit() else 0)
#
#
#     # Use list comprehension to filter out the unwanted features
#
#     # features_to_keep = ["passengers_up", "trip_id_unique_station",
#     #                     "passengers_continue", "station_index", "real_minute_arrival", "arrival_is_estimated",
#     #                     "arrival_hour"]
#     # feature_names = X.columns.tolist()
#     # feature_names = [feature for feature in feature_names if feature not in features_to_keep]
#     # X = X.drop(columns=feature_names)
#
#
#     y_preprocessed = None
#     if "passengers_up" in X.columns:
#         X_preprocessed = X.drop(columns=["passengers_up", "trip_id_unique_station"])
#     else:
#         X_preprocessed = X.drop(columns=["trip_id_unique_station"])
#     if "passengers_up" in X.columns:
#         y_preprocessed = X["passengers_up"]
#     trip_id_unique_station = X["trip_id_unique_station"]
#
#     return X_preprocessed, y_preprocessed, trip_id_unique_station



# import csv
# import numpy as np
# from tqdm import tqdm
# import pandas as pd
# from datetime import datetime
#
#
# def is_time_after(time1, time2, time_format='%H:%M:%S'):
#     """
#     Check if time1 is later or equal to time2.
#
#     Args:
#         time1 (str): First time in string format.
#         time2 (str): Second time in string format.
#         time_format (str): Format of the time string, default is '%H:%M:%S'.
#
#     Returns:
#         bool: True if time1 is later or equal to time2, False otherwise.
#     """
#     try:
#         # Convert time to string if it's not already a string
#         if not isinstance(time1, str):
#             time1 = str(time1)
#         if not isinstance(time2, str):
#             time2 = str(time2)
#
#         t1 = datetime.strptime(time1, time_format)
#         t2 = datetime.strptime(time2, time_format)
#         # return t1 >= t2
#         return t1 > t2
#     except (ValueError, TypeError):
#         return False  # In case of invalid time format or type error
#
#
# def is_time_format(time_string):
#     """
#     Check if the string is a valid time format (HH:MM:SS).
#
#     Args:
#         time_string (str): Time string to validate.
#
#     Returns:
#         bool: True if the string is in correct time format, False otherwise.
#     """
#     try:
#         datetime.strptime(time_string, '%H:%M:%S')
#         return True
#     except ValueError:
#         return False
#
#
# def sum_list(lst):
#     """
#     Sum the number of valid outlier flags in a list.
#
#     Args:
#         lst (list): List of outlier flags (1 for outlier, 0 for valid data).
#
#     Returns:
#         int: Total number of outliers in the list.
#     """
#     return sum(1 for item in lst if item)  # Count non-zero (outlier) values
#
#
# def time_difference(time1, time2, time_format='%H:%M:%S'):
#     """
#     Calculate the time difference in seconds between time1 and time2.
#
#     Args:
#         time1 (str): First time in string format.
#         time2 (str): Second time in string format.
#         time_format (str): Format of the time string, default is '%H:%M:%S'.
#
#     Returns:
#         int: Time difference in seconds, or None if time1 or time2 is invalid.
#     """
#     try:
#         # Ensure both times are strings
#         if not isinstance(time1, str) or pd.isna(time1):
#             return 0  # Return 0 if time1 is invalid
#         if not isinstance(time2, str) or pd.isna(time2):
#             return 0  # Return 0 if time2 is invalid
#
#         t1 = datetime.strptime(time1, time_format)
#         t2 = datetime.strptime(time2, time_format)
#         delta = t2 - t1
#         return int(delta.total_seconds())
#     except (ValueError, TypeError):
#         # Return 0 if there's an issue with the format or type
#         return 0
#
#
# # Mapping for parts and clusters
# part_dict = {"א": "a", "ב": "b", "ג": "c"}
# cluster_dict = {
#     "אונו-אלעד": "Ono-Elad", "בת ים-רמת גן": "Bat yam - Ramat gan",
#     "דרומי-בת ים": "Dromi-Bat yam", 'דרומי-ראשל"צ-חולון': "Dromi-Rashlatz_Holon",
#     'השרון': 'Hasharon', 'חולון עירוני ומטרופוליני+תחרות חולון': "Holon Metropolin",
#     'מזרחי-בני ברק': 'Mizrahi-Bnei brak', 'מזרחי-רמת גן': "Mizrahi-Ramat gan",
#     'פ"ת-ת"א': "Pat-TLV", 'שרון חולון מרחבי': 'Sharon-Holon-Merhavi', 'תל אביב': "TLV"
# }
#
# # Constants for detecting outliers
# THERE_IS_OUTLIAR = 1
# THERE_IS_NOT_OUTLIAR = 0
#
# # Geographic boundaries for valid lat/lon
# LAT_MAX = 33.3
# LAT_MIN = 29.5
# LON_MAX = 35.9
# LON_MIN = 34.2
#
#
# def find_outliars(path_to_dataset):
#     """
#     Identify outliers in the dataset based on a series of conditions and write the results to a CSV file.
#
#     Args:
#         path_to_dataset (str): Path to the dataset CSV file.
#
#     Output:
#         Writes a file 'outliars_rows.csv' with columns indicating the validity of different features.
#     """
#     df = pd.read_csv(path_to_dataset, encoding="ISO-8859-8")
#     with open('outliars_rows.csv', 'w', newline='') as outfile_csv:
#         writer = csv.writer(outfile_csv)
#
#         # Header for the outliers file
#         header = [
#             "trip_id_unique_concat", "trip_id_unique_station_concat", "valid_lat", "valid_lon",
#             "nipuch_is_mult", "trip_id_int", "line_id_int", "direction_is_1_2",
#             "station_index_int", "station_id_int", "arr_time_format", "close_time_format",
#             "close_after_arr", "arr_is_estimate_bool", "passenger_up_is_int_pos",
#             "passenger_cont_is_int_pos", "outliar_sum"
#         ]
#         writer.writerow(header)
#
#         # Iterate over rows and check for outliers
#         for index, row in df.iterrows():
#             list_to_write = []
#
#             # Validate trip_id_unique concatenation
#             check_concat = str(row['trip_id']) + part_dict.get(row['part'], "")
#             list_to_write.append(THERE_IS_NOT_OUTLIAR if check_concat == row['trip_id_unique'] else THERE_IS_OUTLIAR)
#
#             # Validate trip_id_unique_station concatenation
#             check_concat += str(row['station_index'])
#             list_to_write.append(
#                 THERE_IS_NOT_OUTLIAR if check_concat == row['trip_id_unique_station'] else THERE_IS_OUTLIAR)
#
#             # Latitude and Longitude validity
#             list_to_write.append(THERE_IS_NOT_OUTLIAR if LAT_MIN < row["latitude"] < LAT_MAX else THERE_IS_OUTLIAR)
#             list_to_write.append(THERE_IS_NOT_OUTLIAR if LON_MIN < row["longitude"] < LON_MAX else THERE_IS_OUTLIAR)
#
#             # Check nipuch calculation
#             expected_menupach = round(row["passengers_continue"] * row["mekadem_nipuach_luz"], 2)
#             list_to_write.append(
#                 THERE_IS_NOT_OUTLIAR if expected_menupach == row["passengers_continue_menupach"] else THERE_IS_OUTLIAR)
#
#             # Check integer columns
#             for col in ["trip_id", "line_id", "station_index", "station_id"]:
#                 list_to_write.append(THERE_IS_NOT_OUTLIAR if isinstance(row[col], int) else THERE_IS_OUTLIAR)
#
#             # Check direction validity
#             list_to_write.append(THERE_IS_NOT_OUTLIAR if row["direction"] in [1, 2] else THERE_IS_OUTLIAR)
#
#             # Validate time formats
#             list_to_write.append(THERE_IS_NOT_OUTLIAR if is_time_format(row["arrival_time"]) else THERE_IS_OUTLIAR)
#
#             if pd.isna(row["door_closing_time"]):
#                 list_to_write.append("NO DATA")
#                 list_to_write.append("NO DATA")
#             else:
#                 list_to_write.append(
#                     THERE_IS_NOT_OUTLIAR if is_time_format(row["door_closing_time"]) else THERE_IS_OUTLIAR)
#                 list_to_write.append(THERE_IS_NOT_OUTLIAR if is_time_after(row["door_closing_time"],
#                                                                            row["arrival_time"]) else THERE_IS_OUTLIAR)
#
#             # Check boolean validity and integer positivity
#             list_to_write.append(
#                 THERE_IS_NOT_OUTLIAR if isinstance(row["arrival_is_estimated"], bool) else THERE_IS_OUTLIAR)
#             list_to_write.append(THERE_IS_NOT_OUTLIAR if isinstance(row["passengers_up"], int) and row[
#                 "passengers_up"] >= 0 else THERE_IS_OUTLIAR)
#             list_to_write.append(THERE_IS_NOT_OUTLIAR if isinstance(row["passengers_continue"], int) and row[
#                 "passengers_continue"] >= 0 else THERE_IS_OUTLIAR)
#
#             # Calculate sum of outliers
#             list_to_write.append(sum_list(list_to_write))
#
#             writer.writerow(list_to_write)
#
#
# if __name__ == '__main__':
#     """
#     Main function to preprocess the dataset by cleaning and enriching the data,
#     and saving the processed dataset to a CSV file.
#     """
#     # Load the dataset
#     df = pd.read_csv(r'C:\Users\PC\Documents\Year3SemesterB\67577IML\Hackathon\data\HU.BER\train_bus_schedule.csv',
#                      encoding="ISO-8859-8")
#
#     # Drop unnecessary columns
#     df = df.drop(["trip_id_unique", "station_name", "latitude", "longitude"], axis=1)
#     df['time_in_station (sec)'] = 0
#
#     # Fill missing 'arrival_time', 'door_closing_time' and 'door_closing_time' with a default value (if needed)
#     df['arrival_time'].fillna('00:00:00', inplace=True)
#     df["door_closing_time"] = df["door_closing_time"].fillna(df["arrival_time"])
#     df['time_in_station (sec)'].fillna(0, inplace=True)
#
#     # Process the 'alternative' column
#     df['alternative'] = df['alternative'].apply(lambda x: int(x) if x.isdigit() else 0)
#
#     # Iterate over the rows to preprocess data
#     for index, row in tqdm(df.iterrows(), total=df.shape[0]):
#         # Replace part and cluster values using dictionaries
#         df.at[index, 'part'] = part_dict.get(row['part'], row['part'])
#         df.at[index, 'cluster'] = cluster_dict.get(row['cluster'], row['cluster'])
#
#         # Ensure door closing time is after arrival time
#         if not is_time_after(row["door_closing_time"], row["arrival_time"]):
#             df.iloc[index, df.columns.get_loc("door_closing_time")], df.iloc[index, df.columns.get_loc("arrival_time")] = row["arrival_time"], row["door_closing_time"]
#
#         # Set negative passenger_continue values to 0
#         if row["passengers_continue"] <= 0:
#             df.at[index, 'passengers_continue'] = 0
#
#         # Calculate the time spent in the station
#         df.at[index, 'time_in_station (sec)'] = max(time_difference(row["arrival_time"], row["door_closing_time"]), 0)
#
#     # Save the preprocessed dataset
#     df.to_csv(r'C:\Users\PC\Documents\Year3SemesterB\67577IML\Hackathon\data\HU.BER\train_bus_schedule_filtered.csv', index=False)
#     print("Preprocessing complete. Saved to train_bus_schedule_filtered.csv.")


# import csv
# import numpy as np
# from tqdm import tqdm
#
# import pandas as pd
# from datetime import datetime
# def is_time_after(time1, time2, time_format='%H:%M:%S'):
#     try:
#         t1 = datetime.strptime(time1, time_format)
#         t2 = datetime.strptime(time2, time_format)
#         return t1 >= t2
#     except ValueError:
#         return False  # In case of invalid time format
# def is_time_format(time_string):
#     try:
#         datetime.strptime(time_string, '%H:%M:%S')
#         return True
#     except ValueError:
#         return False
#
# def sum_list(lst):
#     total = 0
#     for item in lst:
#         if isinstance(item, int):
#             total += item
#         elif isinstance(item, str):
#             total += 1
#     return total
#
# def time_difference(time1, time2, time_format='%H:%M:%S'):
#     t1 = datetime.strptime(time1, time_format)
#     t2 = datetime.strptime(time2, time_format)
#     delta = t2 - t1
#     return int(delta.total_seconds())
#
# part_dict = {"א": "a", "ב": "b", "ג": "c"}
#
# cluster_dict = {"אונו-אלעד": "Ono-Elad", "בת ים-רמת גן": "Bat yam - Ramat gan", "דרומי-בת ים": "Dromi-Bat yam",
#                 'דרומי-ראשל"צ-חולון': "Dromi-Rashlatz_Holon", 'השרון': 'Hasharon',
#                 'חולון עירוני ומטרופוליני+תחרות חולון': "Holon Metropolin", 'מזרחי-בני ברק': 'Mizrahi-Bnei brak',
#                 'מזרחי-רמת גן': "Mizrahi-Ramat gan", 'פ"ת-ת"א': "Pat-TLV", 'שרון חולון מרחבי': 'Sharon-Holon-Merhavi',
#                 'תל אביב': "TLV"}
#
# THERE_IS_OUTLIAR = 1
# THERE_IS_NOT_OUTLIAR = 0
#
# LAT_MAX = 33.3
# LAT_MIN = 29.5
# LON_MAX = 35.9
# LON_MIN = 34.2
#
# def find_outliars(path_to_dataset):
#
#
#     # Replace 'path_to_file.csv' with the actual path to your CSV file
#     df = pd.read_csv(path_to_dataset, encoding="ISO-8859-8")
#     outfile_csv = open('outliars_rows.csv', 'w', newline='')
#     writer = csv.writer(outfile_csv)
#     header = ["trip_id_unique_concat", "trip_id_unique_station_concat", "valid_lat", "valid_lon",
#               "nipuch_is_mult", "trip_id_int", "line_id_int", "direction_is_1_2", "station_index_int",
#               "station_id_int", "arr_time_format", "close_time_format", "close_after_arr", "arr_is_estimate_bool",
#               "passenger_up_is_int_pos", "passenger_cont_is_int_pos", "outliar_sum"]
#
#     writer.writerow(header)
#     for index, row in df.iterrows():
#         list_to_write = []
#
#         # trip_id_unique_concat
#         check_concat = str(row['trip_id']) + part_dict[row['part']]
#         if check_concat == row['trip_id_unique']:
#             list_to_write.append(THERE_IS_NOT_OUTLIAR)
#         else:
#             list_to_write.append(THERE_IS_OUTLIAR)
#
#         # trip_id_unique_station_concat
#         check_concat += str(row['station_index'])
#         if check_concat == row['trip_id_unique_station']:
#             list_to_write.append(THERE_IS_NOT_OUTLIAR)
#         else:
#             list_to_write.append(THERE_IS_OUTLIAR)
#
#         # valid_lat
#         if LAT_MAX > row["latitude"] > LAT_MIN:
#             list_to_write.append(THERE_IS_NOT_OUTLIAR)
#         else:
#             list_to_write.append(THERE_IS_OUTLIAR)
#
#         # valid_lon
#         if LON_MAX > row["longitude"] > LON_MIN:
#             list_to_write.append(THERE_IS_NOT_OUTLIAR)
#         else:
#             list_to_write.append(THERE_IS_OUTLIAR)
#
#         # nipuch_is_mult
#         if round(row["passengers_continue"] * row["mekadem_nipuach_luz"], 2) == row["passengers_continue_menupach"]:
#             list_to_write.append(THERE_IS_NOT_OUTLIAR)
#         else:
#             list_to_write.append(THERE_IS_OUTLIAR)
#
#         # trip_id_int
#         if isinstance(row["trip_id"], int):
#             list_to_write.append(THERE_IS_NOT_OUTLIAR)
#         else:
#             list_to_write.append(THERE_IS_OUTLIAR)
#
#         # line_id_int
#         if isinstance(row["line_id"], int):
#             list_to_write.append(THERE_IS_NOT_OUTLIAR)
#         else:
#             list_to_write.append(THERE_IS_OUTLIAR)
#
#         # direction_is_1_2
#         if row["direction"] == 1 or row["direction"] == 2:
#             list_to_write.append(THERE_IS_NOT_OUTLIAR)
#         else:
#             list_to_write.append(THERE_IS_OUTLIAR)
#
#         # station_index_int
#         if isinstance(row["station_index"], int):
#             list_to_write.append(THERE_IS_NOT_OUTLIAR)
#         else:
#             list_to_write.append(THERE_IS_OUTLIAR)
#
#         # station_id_int
#         if isinstance(row["station_id"], int):
#             list_to_write.append(THERE_IS_NOT_OUTLIAR)
#         else:
#             list_to_write.append(THERE_IS_OUTLIAR)
#
#         # arr_time_format
#         if is_time_format(row["arrival_time"]):
#             list_to_write.append(THERE_IS_NOT_OUTLIAR)
#         else:
#             list_to_write.append(THERE_IS_OUTLIAR)
#
#         # close_time_format
#         if pd.isna(row["door_closing_time"]):
#             list_to_write.append("NO DATA")
#         elif is_time_format(row["door_closing_time"]):
#             list_to_write.append(THERE_IS_NOT_OUTLIAR)
#         else:
#             list_to_write.append(THERE_IS_OUTLIAR)
#
#         # close_after_arr
#         if pd.isna(row["door_closing_time"]):
#             list_to_write.append("NO DATA")
#         elif is_time_after(row["door_closing_time"], row["arrival_time"]):
#             list_to_write.append(THERE_IS_NOT_OUTLIAR)
#         else:
#             list_to_write.append(THERE_IS_OUTLIAR)
#
#         # arr_is_estimate_bool
#         if isinstance(row["arrival_is_estimated"], bool):
#             list_to_write.append(THERE_IS_NOT_OUTLIAR)
#         else:
#             list_to_write.append(THERE_IS_OUTLIAR)
#
#         # arr_is_estimate_bool
#         if isinstance(row["passengers_up"], int) and row["passengers_up"] >= 0:
#             list_to_write.append(THERE_IS_NOT_OUTLIAR)
#         else:
#             list_to_write.append(THERE_IS_OUTLIAR)
#
#         # passenger_cont_is_int_pos
#         if isinstance(row["passengers_continue"], int) and row["passengers_continue"] >= 0:
#             list_to_write.append(THERE_IS_NOT_OUTLIAR)
#         else:
#             list_to_write.append(THERE_IS_OUTLIAR)
#
#         # outliar_sum
#         list_to_write.append(sum_list(list_to_write))
#
#         writer.writerow(list_to_write)
#
#
# if __name__ == '__main__':
#
#     # Replace 'path_to_file.csv' with the actual path to your CSV file
#     df = pd.read_csv(r'C:\Users\PC\Documents\Year3SemesterB\67577IML\Hackathon\data\train_bus_schedule.csv',
#                      encoding="ISO-8859-8")
#     df = df.drop("station_name", axis=1)
#     df['time_in_station (sec)'] = 0
#
#
#     for index, row in tqdm(df.iterrows(), total=df.shape[0]):
#
#         df.iloc[index, df.columns.get_loc("part")] = part_dict[row["part"]]
#         df.iloc[index, df.columns.get_loc("cluster")] = cluster_dict[row["cluster"]]
#
#         if pd.isna(row["door_closing_time"]):
#             df.iloc[index, df.columns.get_loc("door_closing_time")] = row["arrival_time"]
#             row["door_closing_time"] = row["arrival_time"]
#
#         if not is_time_after(row["door_closing_time"], row["arrival_time"]):
#             df.iloc[index, df.columns.get_loc("door_closing_time")], df.iloc[index, df.columns.get_loc("arrival_time")] = row["arrival_time"], row[
#                 "door_closing_time"]
#
#
#         if pd.isna(row["door_closing_time"]):
#             df.iloc[index, df.columns.get_loc("door_closing_time")] = row["arrival_time"]
#             row["door_closing_time"] = row["arrival_time"]
#
#
#         # close_after_arr
#         if not is_time_after(row["door_closing_time"], row["arrival_time"]):
#             df.iloc[index, df.columns.get_loc("door_closing_time")], df.iloc[index, df.columns.get_loc("arrival_time")] = row["arrival_time"], row[
#                 "door_closing_time"]
#
#
#         # passenger_cont_is_int_pos
#         if row["passengers_continue"] <= 0:
#             df.iloc[index, df.columns.get_loc("passengers_continue")] = 0
#
#         df.iloc[index, df.columns.get_loc('time_in_station (sec)')] = time_difference(row["arrival_time"], row["door_closing_time"])
#
#         # if index == 100:
#         #     break
#
#     # first_50_rows = df.head(50)
#
#     df.to_csv('C:\\Users\\PC\\Documents\\Year3SemesterB\\67577IML\\Hackathon\\data\\train_bus_schedule_filtered.csv', index=False)
#
#
#     # Display the first few rows of the DataFrame
#     # print(df.head())
#
