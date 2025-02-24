from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor  # Robust linear model
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub
import os
from data import preprocess, create_preprocessor

# load data
path = kagglehub.dataset_download("stoney71/new-york-city-transport-statistics")
csv_file = os.path.join(path, 'mta_1706.csv')
df = pd.read_csv(csv_file, on_bad_lines='skip')

# preprocess
X, y = preprocess(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create and train pipeline
pipeline = make_pipeline(
    create_preprocessor(),
    StandardScaler(with_mean=False),  # Handles sparse one-hot encoded features
    HuberRegressor(
        epsilon=1.35,  # outlier sensitivity
        alpha=0.0001,  # regularization strength
        max_iter=1000
    )
)

pipeline.fit(X_train, y_train)

# save pipeline
joblib.dump(pipeline, 'delay_pipeline.pkl')

# evaluate
preds = pipeline.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, preds):.2f} minutes")

# plot residuals
residuals = y_test - preds
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(0, color='red', linestyle='dashed', linewidth=2)
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Residual Histogram")
plt.show()

# plot expected vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, preds, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)  # Diagonal line

plt.axis("equal")
plt.xlabel('Actual Delay (minutes)')
plt.ylabel('Predicted Delay (minutes)')
plt.title('Actual vs. Predicted Delay')
plt.show()
