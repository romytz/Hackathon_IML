# Import necessary libraries
import datetime  # Used for handling and calculating time differences
import pandas as pd  # Used for data manipulation and analysis


def create_duration_column(data_path):
    """
    Creates a 'duration' column in the dataset, representing the trip duration in minutes for each unique trip.

    Parameters:
    - data_path: The file path to the CSV file containing the bus schedule data.

    This function reads the data, calculates the duration (in minutes) for each unique trip based on 
    the difference between the earliest and latest 'arrival_time' entries per 'trip_id_unique' group, 
    and saves the modified dataset to a new CSV file.
    """
    # Load the data from the specified CSV file with ISO-8859-8 encoding
    data = pd.read_csv(data_path, encoding="ISO-8859-8")

    # Convert 'arrival_time' column to datetime.time format (only time, no date)
    data["arrival_time"] = pd.to_datetime(data["arrival_time"], format='%H:%M:%S').dt.time

    # Group data by unique trip IDs
    data_by_group = data.groupby('trip_id_unique')

    # Calculate duration for each trip as the time difference in minutes between the first and last arrival times
    data['duration'] = data_by_group['arrival_time'].transform(
        lambda x: (datetime.datetime.combine(datetime.date.today(), x.max())  # Latest arrival time
                   - datetime.datetime.combine(datetime.date.today(), x.min())  # Earliest arrival time
                   ).total_seconds() / 60  # Convert seconds to minutes
    )

    # Save the updated DataFrame to a new CSV file with the added 'duration' column
    data.to_csv("data/train_bus_schedule_duration.csv", index=False, encoding="ISO-8859-8")
