from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from data import preprocess
import kagglehub
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.robust.robust_linear_model import RLM
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt



# Loading and reading data
path = kagglehub.dataset_download("stoney71/new-york-city-transport-statistics")

print("Path to dataset files:", path)
print("Files in the directory:", os.listdir(path))

csv_file = os.path.join(path, 'mta_1706.csv')  # Adjust file name accordingly

df = pd.read_csv(csv_file, on_bad_lines='skip')

x, y = preprocess(df)
X = sm.add_constant(x)
rlm_model = RLM(y.astype(float), X.astype(float), M = sm.robust.norms.HuberT())
rlm_results = rlm_model.fit()
y_pred = rlm_results.predict(X)

residuals = y - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='red', linewidth=2)
plt.xlabel('Actual Delay (minutes)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residuals vs Actual Delay')
plt.show()


# 80% for training, 20% for testing
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
#y_pred = model.predict(x_train)

"""
sample = np.random.randint(0, len(x_train), size=10000)

x_train_sample = x_train.iloc[sample]
y_train_sample = y_train.iloc[sample]
"""
"""
#model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
naive_model = LinearRegression().fit(x_train, y_train)
y_naive_pred = naive_model.predict(x_train)

naive_residuals = y_train - y_naive_pred
bias_model = LinearRegression().fit(y_naive_pred.reshape(-1, 1), naive_residuals)


# testing predict
y_test_naive = naive_model.predict(x_test)
y_test_pred = y_test_naive - bias_model.predict(y_test_naive.reshape(-1, 1))

bias_residuals = bias_model.predict(naive_residuals.to_frame())

#plt.scatter(y_test, y_test - y_test_pred, alpha= 0.5)

plt.scatter(naive_residuals, bias_residuals)
plt.show()

print('done predicting')


# performance metrics
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
"""
"""
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
"""
"""
#residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_test, residuals, alpha=0.5)
plt.axhline(0, color='red', linewidth=2)
plt.xlabel('Actual Delay (minutes)')
plt.ylabel('Residual (Actual - Predicted)')
plt.title('Residuals vs Actual Delay')
plt.show()
"""
"""
#Actual vs. Predicted Delay Scatterplot
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y_pred.min(), y_pred.max()], color='red', linewidth=2)
plt.xlabel('Actual Delay (minutes)')
plt.ylabel('Predicted Delay (minutes)')
plt.title('Actual vs Predicted Delay')
plt.show()


plt.hist(y_test, bins=50)
plt.xlabel('Actual Delay')
plt.ylabel('Frequency')
plt.title('Histogram of Actual Delays')
plt.show()"""

"""
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.xlabel('Actual Delay (minutes)')
plt.ylabel('Predicted Delay (minutes)')
plt.title('Actual vs Predicted Delay')
plt.show()
"""