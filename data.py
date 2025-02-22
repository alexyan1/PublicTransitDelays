import kagglehub
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("stoney71/new-york-city-transport-statistics")


print("Path to dataset files:", path)
"""
df = pd.read_csv('mta_1706.csv', on_bad_lines= 'skip')


# Drop rows with missing values in the critical columns
df = df.dropna(subset=['ScheduledArrivalTime', 'RecordedAtTime'])

# Convert scheduled/actual arrival times to datetime format
df['ScheduledArrivalTime'] = pd.to_datetime(df['ScheduledArrivalTime'])
df['RecordedAtTime'] = pd.to_datetime(df['RecordedAtTime'])

# Calculate delay in minutes
df['Delay'] = (df['RecordedAtTime'] - df['ScheduledArrivalTime']).dt.total_seconds() / 60

# Additional data
df['Hour'] = df['ScheduledArrivalTime'].dt.hour
df['DayOfWeek'] = df['ScheduledArrivalTime'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)  # 5, 6 represent Saturday and Sunday

print(df.head(3))

# print(df.columns)
"""