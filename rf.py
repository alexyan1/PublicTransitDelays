from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
import joblib
import pandas as pd
import os
import kagglehub
from data import preprocess, create_preprocessor

# Load and preprocess data
path = kagglehub.dataset_download("stoney71/new-york-city-transport-statistics")
csv_file = os.path.join(path, 'mta_1710.csv')
df = pd.read_csv(csv_file, on_bad_lines='skip')

X, y = preprocess(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train pipeline
pipeline = make_pipeline(
    create_preprocessor(),
    RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
)

pipeline.fit(X_train, y_train)

# Save entire pipeline
joblib.dump(pipeline, 'transport_delay_pipeline.pkl')

# Evaluate
preds = pipeline.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, preds):.2f} minutes")