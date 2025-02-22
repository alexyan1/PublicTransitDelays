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

# Drop rows with missing values in the critical columns
df = df.dropna(subset=['ScheduledArrivalTime', 'RecordedAtTime'])

# Assuming your 'ScheduledArrivalTime' column has invalid '24:00' times
df['ScheduledArrivalTime'] = df['ScheduledArrivalTime'].replace(to_replace="24:00", value="00:00")
df['ScheduledArrivalTime'] = df['ScheduledArrivalTime'].replace(to_replace="25:00", value="01:00")

# Convert the 'ScheduledArrivalTime' to datetime after replacement
df['ScheduledArrivalTime'] = pd.to_datetime(df['ScheduledArrivalTime'], errors='coerce')

df['RecordedAtTime'] = pd.to_datetime(df['RecordedAtTime'])

# Additional data
df['Hour'] = df['ScheduledArrivalTime'].dt.hour
df['DayOfWeek'] = df['ScheduledArrivalTime'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)  # 5, 6 represent Saturday and Sunday

print(df.head(3))

# print(df.columns)
