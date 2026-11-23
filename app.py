from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load Model
import os
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "traffic_forecasting_lstm.keras")

model = load_model(model_path)

# Load Scaler
with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(BASE_DIR, "feature_columns.pkl"), "rb") as f:
    feature_columns = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    temp = float(request.form["temp"])
    rain = float(request.form["rain"])
    snow = float(request.form["snow"])
    clouds = int(request.form["clouds"])
    hour = int(request.form["hour"])
    month = int(request.form["month"])
    day = int(request.form["day"])
    year = int(request.form["year"])

    input_df = pd.DataFrame([{
        "temp": temp,
        "rain_1h": rain,
        "snow_1h": snow,
        "clouds_all": clouds,
        "hour": hour,
        "day": day,
        "month": month,
        "year": year
    }])

    # Add missing one-hot encoded columns
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_columns]

    input_scaled = scaler.transform(input_df)

    # Create LSTM input
    input_scaled = np.repeat(input_scaled[np.newaxis,:,:],24,axis=1)

    prediction = model.predict(input_scaled)

    return render_template(
        "index.html",
        prediction_text=f"Predicted Traffic Volume : {prediction[0][0]:.2f}"
    )


if __name__ == "__main__":
    app.run(debug=True)