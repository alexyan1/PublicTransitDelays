import kagglehub
import pandas as pd
df = pd.read_csv('mta_1706.csv', on_bad_lines= 'skip')

print(df.head(3))
