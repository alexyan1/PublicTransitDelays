import kagglehub
import os
import pandas as pd

# adjust invalid hours that are greater than 23
def adjust_times(df):
    # convert RecordedAtTime to datetime and extract date
    df['RecordedAtTime'] = pd.to_datetime(df['RecordedAtTime'], errors='coerce')
    recorded_dates = df['RecordedAtTime'].dt.normalize()  # Get date part
    
    # helper function to process time columns
    def process_time(col):
        # extract time components from string
        time_parts = col.str.extract(r'(\d{1,2}):(\d{1,2}):?(\d{0,2})')
        
        # convert to numeric with error handling
        hours = pd.to_numeric(time_parts[0], errors='coerce') % 24
        minutes = pd.to_numeric(time_parts[1], errors='coerce').fillna(0)
        seconds = pd.to_numeric(time_parts[2], errors='coerce').fillna(0)
        
        # combine with recorded dates
        return recorded_dates + pd.to_timedelta(hours, unit='h') \
                              + pd.to_timedelta(minutes, unit='m') \
                              + pd.to_timedelta(seconds, unit='s')

    # process both time columns
    df['ScheduledArrivalTime'] = process_time(df['ScheduledArrivalTime'])
    df['ExpectedArrivalTime'] = process_time(df['ExpectedArrivalTime'])
    
    return df

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
    df = df.sample(n=10000, random_state=42)
    df = adjust_times(df)
    df = df.dropna(subset=['ScheduledArrivalTime', 'RecordedAtTime'])
    
    df['ScheduledArrivalTime'] = pd.to_datetime(df['ScheduledArrivalTime'], errors='coerce')
    
    # apply the delay calculation to each row
    df['Delay'] = df.apply(calculate_delay, axis=1)
    
    # Hour, DayOfWeek, and IsWeekend columns
    df['Hour'] = df['ScheduledArrivalTime'].dt.hour
    df['DayOfWeek'] = df['ScheduledArrivalTime'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    # drop rows with missing delay values
    df = df.dropna(subset=['Delay'])
    
    # one-hot encode PublishedLineName and NextStopName
    df = pd.get_dummies(df, columns=['PublishedLineName', 'NextStopPointName', 'DayOfWeek'], drop_first=True)
    
    # select features and target (include the new encoded columns)
    features = ['Hour', 'DistanceFromStop'] + \
               [col for col in df.columns if col.startswith('PublishedLineName_') or col.startswith('NextStopName_')]
    x = df[features]
    y = df['Delay']
    print(df['Delay'].describe())
    
    return x, y

if "__name__" == "__main__":
    # Download latest version
    path = kagglehub.dataset_download("stoney71/new-york-city-transport-statistics")
    # print("Path to dataset files:", path)
    # print("Files in the directory:", os.listdir(path))

    csv_file = os.path.join(path, 'mta_1706.csv')  # Adjust file name accordingly
    print("loading csv")
    df = pd.read_csv(csv_file, on_bad_lines='skip').head(30)