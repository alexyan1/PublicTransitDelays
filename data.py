import kagglehub
import os
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("stoney71/new-york-city-transport-statistics")

print("Path to dataset files:", path)
print("Files in the directory:", os.listdir(path))

csv_file = os.path.join(path, 'mta_1706.csv')  # Adjust file name accordingly

df = pd.read_csv(csv_file, on_bad_lines='skip')

# df = df.head(30)
# print(df.head())

# adjust invalid hours that are greater than 23
def adjust_invalid_hour_format(time_str):
    if isinstance(time_str, str):
        hour = int(time_str.split(":")[0])  # Extract the hour part
        if hour >= 24:
            # Calculate the valid hour by subtracting 24 and adjusting the day
            new_hour = hour - 24
            time_str = time_str.replace(f'{hour}:', f'{new_hour}:', 1)  # Replace hour with new hour
            time_obj = pd.to_datetime(time_str)  # Convert to datetime
            time_obj += pd.Timedelta(days=1)  # Add one day
            return time_obj
        return pd.to_datetime(time_str)  # Return as is for valid times
    return pd.NaT  # Return NaT for non-string entries (e.g., NaN)

def adjust_times(df):
    # set the same date (e.g., 01-01-2000) for 'ScheduledArrivalTime' and 'RecordedAtTime'
    def normalize_date(time_str):
        time_obj = pd.to_datetime(time_str, errors='coerce')
        if pd.notna(time_obj):
            return time_obj.replace(year=2000, month=1, day=1)
        return pd.NaT   # invalid date

    df['ScheduledArrivalTime'] = df['ScheduledArrivalTime'].apply(normalize_date)
    df['ExpectedArrivalTime'] = df['ExpectedArrivalTime'].apply(normalize_date)
    df['RecordedAtTime'] = df['RecordedAtTime'].apply(normalize_date)

    return df

def calculate_delay(row):
    # determine whether to use RecordedAtTime or ExpectedArrivalTime
    if row['ArrivalProximityText'] == 'at stop':
        arrival_time = row['RecordedAtTime']
    else:
        arrival_time = row['ExpectedArrivalTime']

    # handle cases where arrival_time appears to be before scheduled time due to crossing midnight
    if arrival_time < row['ScheduledArrivalTime']:
        arrival_time += pd.Timedelta(days=1)

    # calculate delay in minutes
    delay = (arrival_time - row['ScheduledArrivalTime']).total_seconds() / 60
    return delay

def preprocess(df):
    df = adjust_times(df)
    df = df.dropna(subset=['ScheduledArrivalTime', 'RecordedAtTime'])

    # Apply the delay calculation to each row
    df['Delay'] = df.apply(calculate_delay, axis=1)

    # Additional data: Hour, DayOfWeek, and IsWeekend columns
    df['Hour'] = df['ScheduledArrivalTime'].dt.hour
    df['DayOfWeek'] = df['ScheduledArrivalTime'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)  # 5, 6 represent Saturday and Sunday

    # Check the first few rows
    print(df.head())
    print(df[['ScheduledArrivalTime', 'ExpectedArrivalTime', 'RecordedAtTime', 'Delay']].head())

    # Select features and target
    x = df[['Hour', 'DistanceFromStop']]  # Add other relevant features
    y = df['Delay']

    return x, y