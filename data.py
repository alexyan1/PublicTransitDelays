import kagglehub
import os
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("stoney71/new-york-city-transport-statistics")
print("Path to dataset files:", path)

print("Files in the directory:", os.listdir(path))

csv_file = os.path.join(path, 'mta_1706.csv')  # Adjust file name accordingly

df = pd.read_csv(csv_file, on_bad_lines='skip')
df = df.head(30)

print(df.head())

# Drop rows with missing values in the critical columns
df = df.dropna(subset=['ScheduledArrivalTime', 'RecordedAtTime'])

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

# Vectorized approach: Find rows where the hour is greater than or equal to 24
def adjust_times(df):
    # Set the same date (e.g., 01-01-2000) for both 'ScheduledArrivalTime' and 'RecordedAtTime'
    def normalize_date(time_str):
        # Convert to datetime
        time_obj = pd.to_datetime(time_str, errors='coerce')
        # If valid, replace year, month, day to be constant (01-01-2000)
        if pd.notna(time_obj):
            return time_obj.replace(year=2000, month=1, day=1)
        return pd.NaT

    df['ScheduledArrivalTime'] = df['ScheduledArrivalTime'].apply(normalize_date)
    df['RecordedAtTime'] = df['RecordedAtTime'].apply(normalize_date)

    return df

# Apply the function to adjust the times
df = adjust_times(df)

# Drop rows with invalid 'ScheduledArrivalTime' or 'RecordedAtTime' after conversion
df = df.dropna(subset=['ScheduledArrivalTime', 'RecordedAtTime'])

# Calculate the delay, taking into account the possibility of a day crossover
def calculate_delay(row):
    # If 'RecordedAtTime' is earlier in the day than 'ScheduledArrivalTime', add a full day to RecordedAtTime
    if row['RecordedAtTime'] < row['ScheduledArrivalTime']:
        row['RecordedAtTime'] += pd.Timedelta(days=1)
    
    # Calculate the delay in minutes
    delay = (row['RecordedAtTime'] - row['ScheduledArrivalTime']).total_seconds() / 60
    return delay

# Apply the delay calculation to each row
df['Delay'] = df.apply(calculate_delay, axis=1)

# Additional data: Hour, DayOfWeek, and IsWeekend columns
df['Hour'] = df['ScheduledArrivalTime'].dt.hour
df['DayOfWeek'] = df['ScheduledArrivalTime'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)  # 5, 6 represent Saturday and Sunday

# Check the first few rows
print(df.head())
print(df[['ScheduledArrivalTime', 'RecordedAtTime', 'Delay']].head())
