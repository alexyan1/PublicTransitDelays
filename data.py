import kagglehub
import os
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("stoney71/new-york-city-transport-statistics")
print("Path to dataset files:", path)

print("Files in the directory:", os.listdir(path))

csv_file = os.path.join(path, 'mta_1706.csv')  # Adjust file name accordingly

df = pd.read_csv(csv_file, on_bad_lines='skip')

print(df.head())

# drop rows with missing values in the critical columns

# Drop rows with missing values in the critical columns
df = df.dropna(subset=['ScheduledArrivalTime', 'RecordedAtTime'])

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
    # Identify the rows where the hour is 24 or greater
    condition = df['ScheduledArrivalTime'].str.split(':').str[0].astype(int) >= 24

    # Apply the adjustment only to those rows
    df.loc[condition, 'ScheduledArrivalTime'] = df.loc[condition, 'ScheduledArrivalTime'].apply(adjust_invalid_hour_format)

    # Convert the remaining valid times to datetime
    df['ScheduledArrivalTime'] = pd.to_datetime(df['ScheduledArrivalTime'], errors='coerce')

    return df

# Apply the adjustment to the entire DataFrame
df = adjust_times(df)

# Check the first few rows
print(df.head())