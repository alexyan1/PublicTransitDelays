from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from data import preprocess
import kagglehub
import os
import pandas as pd

# Split data into training and testing sets (80% training, 20% testing)

# Loading and reading data
path = kagglehub.dataset_download("stoney71/new-york-city-transport-statistics")

print("Path to dataset files:", path)
print("Files in the directory:", os.listdir(path))

csv_file = os.path.join(path, 'mta_1706.csv')  # Adjust file name accordingly

df = pd.read_csv(csv_file, on_bad_lines='skip')

x, y = preprocess(df)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_pred.min(), y_pred.max()], color='red', linewidth=2)
plt.xlabel('Actual Delay (minutes)')
plt.ylabel('Predicted Delay (minutes)')
plt.title('Actual vs Predicted Delay')
plt.show()