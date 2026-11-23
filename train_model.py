import os
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

# Project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Generate sample traffic data
np.random.seed(42)

n = 6000

data = pd.DataFrame({
    "temp": np.random.uniform(0, 40, n),
    "rain_1h": np.random.uniform(0, 10, n),
    "snow_1h": np.random.uniform(0, 2, n),
    "clouds_all": np.random.randint(0, 101, n),
    "hour": np.random.randint(0, 24, n),
    "day": np.random.randint(1, 29, n),
    "month": np.random.randint(1, 13, n),
    "year": np.random.randint(2020, 2027, n)
})

# Generate demonstration traffic volume
rush_hours = data["hour"].isin([7, 8, 9, 16, 17, 18, 19])

traffic = (
    1200
    + rush_hours.astype(int) * 3000
    + (data["clouds_all"] * 3)
    - (data["rain_1h"] * 40)
    - (data["snow_1h"] * 100)
    + np.random.normal(0, 250, n)
)

data["traffic_volume"] = np.clip(traffic, 100, 8000)

# Select input features
feature_columns = [
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "hour",
    "day",
    "month",
    "year"
]

X = data[feature_columns]
y = data["traffic_volume"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Prepare LSTM input (24 time steps)
X_lstm = np.repeat(
    X_scaled[:, np.newaxis, :],
    24,
    axis=1
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_lstm,
    y,
    test_size=0.2,
    random_state=42
)

# Build LSTM model
model = Sequential([
    Input(shape=(24, len(feature_columns))),
    LSTM(32),
    Dense(16, activation="relu"),
    Dropout(0.2),
    Dense(1)
])

# Compile model
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# Train model
print("Training LSTM model...")

model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate model
loss, mae = model.evaluate(X_test, y_test)

print("Model MAE:", mae)

# Save trained model
model.save(
    os.path.join(BASE_DIR, "traffic_forecasting_lstm.keras")
)

# Save scaler
with open(os.path.join(BASE_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# Save feature columns
with open(os.path.join(BASE_DIR, "feature_columns.pkl"), "wb") as f:
    pickle.dump(feature_columns, f)

print("\nAll files generated successfully!")