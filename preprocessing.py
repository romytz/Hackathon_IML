import csv
import numpy as np
from tqdm import tqdm

import pandas as pd
from datetime import datetime
def is_time_after(time1, time2, time_format='%H:%M:%S'):
    try:
        t1 = datetime.strptime(time1, time_format)
        t2 = datetime.strptime(time2, time_format)
        return t1 >= t2
    except ValueError:
        return False  # In case of invalid time format
def is_time_format(time_string):
    try:
        datetime.strptime(time_string, '%H:%M:%S')
        return True
    except ValueError:
        return False

def sum_list(lst):
    total = 0
    for item in lst:
        if isinstance(item, int):
            total += item
        elif isinstance(item, str):
            total += 1
    return total

def time_difference(time1, time2, time_format='%H:%M:%S'):
    t1 = datetime.strptime(time1, time_format)
    t2 = datetime.strptime(time2, time_format)
    delta = t2 - t1
    return int(delta.total_seconds())

part_dict = {"א": "a", "ב": "b", "ג": "c"}

cluster_dict = {"אונו-אלעד": "Ono-Elad", "בת ים-רמת גן": "Bat yam - Ramat gan", "דרומי-בת ים": "Dromi-Bat yam",
                'דרומי-ראשל"צ-חולון': "Dromi-Rashlatz_Holon", 'השרון': 'Hasharon',
                'חולון עירוני ומטרופוליני+תחרות חולון': "Holon Metropolin", 'מזרחי-בני ברק': 'Mizrahi-Bnei brak',
                'מזרחי-רמת גן': "Mizrahi-Ramat gan", 'פ"ת-ת"א': "Pat-TLV", 'שרון חולון מרחבי': 'Sharon-Holon-Merhavi',
                'תל אביב': "TLV"}

THERE_IS_OUTLIAR = 1
THERE_IS_NOT_OUTLIAR = 0

LAT_MAX = 33.3
LAT_MIN = 29.5
LON_MAX = 35.9
LON_MIN = 34.2

def find_outliars(path_to_dataset):


    # Replace 'path_to_file.csv' with the actual path to your CSV file
    df = pd.read_csv(path_to_dataset, encoding="ISO-8859-8")
    outfile_csv = open('outliars_rows.csv', 'w', newline='')
    writer = csv.writer(outfile_csv)
    header = ["trip_id_unique_concat", "trip_id_unique_station_concat", "valid_lat", "valid_lon",
              "nipuch_is_mult", "trip_id_int", "line_id_int", "direction_is_1_2", "station_index_int",
              "station_id_int", "arr_time_format", "close_time_format", "close_after_arr", "arr_is_estimate_bool",
              "passenger_up_is_int_pos", "passenger_cont_is_int_pos", "outliar_sum"]

    writer.writerow(header)
    for index, row in df.iterrows():
        list_to_write = []

        # trip_id_unique_concat
        check_concat = str(row['trip_id']) + part_dict[row['part']]
        if check_concat == row['trip_id_unique']:
            list_to_write.append(THERE_IS_NOT_OUTLIAR)
        else:
            list_to_write.append(THERE_IS_OUTLIAR)

        # trip_id_unique_station_concat
        check_concat += str(row['station_index'])
        if check_concat == row['trip_id_unique_station']:
            list_to_write.append(THERE_IS_NOT_OUTLIAR)
        else:
            list_to_write.append(THERE_IS_OUTLIAR)

        # valid_lat
        if LAT_MAX > row["latitude"] > LAT_MIN:
            list_to_write.append(THERE_IS_NOT_OUTLIAR)
        else:
            list_to_write.append(THERE_IS_OUTLIAR)

        # valid_lon
        if LON_MAX > row["longitude"] > LON_MIN:
            list_to_write.append(THERE_IS_NOT_OUTLIAR)
        else:
            list_to_write.append(THERE_IS_OUTLIAR)

        # nipuch_is_mult
        if round(row["passengers_continue"] * row["mekadem_nipuach_luz"], 2) == row["passengers_continue_menupach"]:
            list_to_write.append(THERE_IS_NOT_OUTLIAR)
        else:
            list_to_write.append(THERE_IS_OUTLIAR)

        # trip_id_int
        if isinstance(row["trip_id"], int):
            list_to_write.append(THERE_IS_NOT_OUTLIAR)
        else:
            list_to_write.append(THERE_IS_OUTLIAR)

        # line_id_int
        if isinstance(row["line_id"], int):
            list_to_write.append(THERE_IS_NOT_OUTLIAR)
        else:
            list_to_write.append(THERE_IS_OUTLIAR)

        # direction_is_1_2
        if row["direction"] == 1 or row["direction"] == 2:
            list_to_write.append(THERE_IS_NOT_OUTLIAR)
        else:
            list_to_write.append(THERE_IS_OUTLIAR)

        # station_index_int
        if isinstance(row["station_index"], int):
            list_to_write.append(THERE_IS_NOT_OUTLIAR)
        else:
            list_to_write.append(THERE_IS_OUTLIAR)

        # station_id_int
        if isinstance(row["station_id"], int):
            list_to_write.append(THERE_IS_NOT_OUTLIAR)
        else:
            list_to_write.append(THERE_IS_OUTLIAR)

        # arr_time_format
        if is_time_format(row["arrival_time"]):
            list_to_write.append(THERE_IS_NOT_OUTLIAR)
        else:
            list_to_write.append(THERE_IS_OUTLIAR)

        # close_time_format
        if pd.isna(row["door_closing_time"]):
            list_to_write.append("NO DATA")
        elif is_time_format(row["door_closing_time"]):
            list_to_write.append(THERE_IS_NOT_OUTLIAR)
        else:
            list_to_write.append(THERE_IS_OUTLIAR)

        # close_after_arr
        if pd.isna(row["door_closing_time"]):
            list_to_write.append("NO DATA")
        elif is_time_after(row["door_closing_time"], row["arrival_time"]):
            list_to_write.append(THERE_IS_NOT_OUTLIAR)
        else:
            list_to_write.append(THERE_IS_OUTLIAR)

        # arr_is_estimate_bool
        if isinstance(row["arrival_is_estimated"], bool):
            list_to_write.append(THERE_IS_NOT_OUTLIAR)
        else:
            list_to_write.append(THERE_IS_OUTLIAR)

        # arr_is_estimate_bool
        if isinstance(row["passengers_up"], int) and row["passengers_up"] >= 0:
            list_to_write.append(THERE_IS_NOT_OUTLIAR)
        else:
            list_to_write.append(THERE_IS_OUTLIAR)

        # passenger_cont_is_int_pos
        if isinstance(row["passengers_continue"], int) and row["passengers_continue"] >= 0:
            list_to_write.append(THERE_IS_NOT_OUTLIAR)
        else:
            list_to_write.append(THERE_IS_OUTLIAR)

        # outliar_sum
        list_to_write.append(sum_list(list_to_write))

        writer.writerow(list_to_write)


if __name__ == '__main__':

    # Replace 'path_to_file.csv' with the actual path to your CSV file
    df = pd.read_csv(r'C:\Elchanan\masters\second_year\Semester_B\IML\hackathon\HU.BER\train_bus_schedule.csv',
                     encoding="ISO-8859-8")
    df = df.drop("station_name", axis=1)
    df['time_in_station (sec)'] = 0


    for index, row in tqdm(df.iterrows(), total=df.shape[0]):

        df.iloc[index, df.columns.get_loc("part")] = part_dict[row["part"]]
        df.iloc[index, df.columns.get_loc("cluster")] = cluster_dict[row["cluster"]]

        if pd.isna(row["door_closing_time"]):
            df.iloc[index, df.columns.get_loc("door_closing_time")] = row["arrival_time"]
            row["door_closing_time"] = row["arrival_time"]

        if not is_time_after(row["door_closing_time"], row["arrival_time"]):
            df.iloc[index, df.columns.get_loc("door_closing_time")], df.iloc[index, df.columns.get_loc("arrival_time")] = row["arrival_time"], row[
                "door_closing_time"]


        if pd.isna(row["door_closing_time"]):
            df.iloc[index, df.columns.get_loc("door_closing_time")] = row["arrival_time"]
            row["door_closing_time"] = row["arrival_time"]


        # close_after_arr
        if not is_time_after(row["door_closing_time"], row["arrival_time"]):
            df.iloc[index, df.columns.get_loc("door_closing_time")], df.iloc[index, df.columns.get_loc("arrival_time")] = row["arrival_time"], row[
                "door_closing_time"]


        # passenger_cont_is_int_pos
        if row["passengers_continue"] <= 0:
            df.iloc[index, df.columns.get_loc("passengers_continue")] = 0

        df.iloc[index, df.columns.get_loc('time_in_station (sec)')] = time_difference(row["arrival_time"], row["door_closing_time"])

        # if index == 100:
        #     break

    # first_50_rows = df.head(50)

    df.to_csv('train_bus_schedule_filtered.csv', index=False)

    # Display the first few rows of the DataFrame
    # print(df.head())

