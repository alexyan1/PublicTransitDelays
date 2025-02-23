import kagglehub
import os
import pandas as pd

# Download latest version

path = kagglehub.dataset_download("stoney71/new-york-city-transport-statistics")
# print("Path to dataset files:", path)
# print("Files in the directory:", os.listdir(path))

csv_file = os.path.join(path, 'mta_1706.csv')  # Adjust file name accordingly

df = pd.read_csv(csv_file, on_bad_lines='skip')

# print(df.head())
print("loading csv")
df = pd.read_csv(csv_file, on_bad_lines='skip').head(30)

# adjust invalid hours that are greater than 23
def adjust_invalid_hour_format(time_str):
    if pd.isna(time_str) or not isinstance(time_str, str):
        return None

    # extract time part if input is a datetime string
    if " " in time_str:
        _, time_part = time_str.split(" ", 1)
    else:
        time_part = time_str

    try:
        parts = time_part.split(":")
        hour = int(parts[0])
        minutes = int(parts[1]) if len(parts) > 1 else 0
        seconds = int(parts[2]) if len(parts) > 2 else 0

        # wrap hours >=24 to 0-23 without adding days
        hour = hour % 24

        return pd.Timestamp.time(
            pd.Timestamp(year=2000, month=1, day=1, hour=hour, minute=minutes, second=seconds)
        )
    except:
        return None

def adjust_times(df):
    def apply_recorded_date(row):
        recorded_time = pd.to_datetime(row["RecordedAtTime"], errors="coerce")
        if pd.isna(recorded_time):
            return row

        # process ScheduledArrivalTime
        s_time = adjust_invalid_hour_format(row["ScheduledArrivalTime"])
        if s_time:
            # use date from RecordedAtTime
            row["ScheduledArrivalTime"] = pd.Timestamp.combine(recorded_time.date(), s_time)
        else:
            row["ScheduledArrivalTime"] = pd.NaT

        # process ExpectedArrivalTime similarly
        e_time = adjust_invalid_hour_format(row["ExpectedArrivalTime"])
        if e_time:
            row["ExpectedArrivalTime"] = pd.Timestamp.combine(recorded_time.date(), e_time)
        else:
            row["ExpectedArrivalTime"] = pd.NaT

        return row

    return df.apply(apply_recorded_date, axis=1)

def calculate_delay(row):
    # clean the ArrivalProximityText to handle whitespace/case issues
    proximity = str(row['ArrivalProximityText']).strip().lower()
    
    # check if ExpectedArrivalTime is NaN and proximity is 'at stop'
    if pd.isna(row['ExpectedArrivalTime']) and proximity == 'at stop':
        arrival_time = pd.to_datetime(row['RecordedAtTime'], errors='coerce')
    else:
        # default logic based on proximity text
        if proximity == 'at stop':
            arrival_time = pd.to_datetime(row['RecordedAtTime'], errors='coerce')
        else:
            arrival_time = pd.to_datetime(row['ExpectedArrivalTime'], errors='coerce')

    scheduled_time = pd.to_datetime(row['ScheduledArrivalTime'], errors='coerce')

    if pd.isna(arrival_time) or pd.isna(scheduled_time):
        return None

    # midnight crossover check (only adjust if >12 hours early)
    time_diff = arrival_time - scheduled_time
    if time_diff.total_seconds() < 0:  # arrival is earlier
        if abs(time_diff) > pd.Timedelta(hours=12):
            arrival_time += pd.Timedelta(days=1)

    delay = (arrival_time - scheduled_time).total_seconds() / 60
    return delay

def preprocess(df):
    df = df.head(30)
    df = adjust_times(df)
    df = df.dropna(subset=['ScheduledArrivalTime', 'RecordedAtTime'])

    # mkae sure ScheduledArrivalTime is in datetime format
    df['ScheduledArrivalTime'] = pd.to_datetime(df['ScheduledArrivalTime'], errors='coerce')

    # apply the delay calculation to each row
    df['Delay'] = df.apply(calculate_delay, axis=1)

    # additional data: Hour, DayOfWeek, and IsWeekend columns
    df['Hour'] = df['ScheduledArrivalTime'].dt.hour
    df['DayOfWeek'] = df['ScheduledArrivalTime'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)  # 5, 6 represent Saturday and Sunday

    df = df.dropna(subset=['Delay'])    # drop NA values

    # check the first few rows
    # print(df.head())
    # print(df[['ScheduledArrivalTime', 'ExpectedArrivalTime', 'RecordedAtTime', 'Delay']].head())

    # select features and target
    x = df[['Hour', 'DistanceFromStop', 'IsWeekend', 'NextStopPointName']]
    y = df['Delay']

    return x, y
