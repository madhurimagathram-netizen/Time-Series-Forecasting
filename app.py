
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import os

from tensorflow.keras.models import load_model

app = Flask(__name__)

# Allow frontend to connect
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model
model = load_model(
    os.path.join(BASE_DIR, "traffic_forecasting_lstm.keras")
)

# Load scaler
with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

# Load feature columns
with open(os.path.join(BASE_DIR, "feature_columns.pkl"), "rb") as f:
    feature_columns = pickle.load(f)


@app.route("/")
def home():
    return jsonify({
        "message": "Traffic Forecasting API is running!"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        input_df = pd.DataFrame([{
            "temp": float(data["temp"]),
            "rain_1h": float(data["rain"]),
            "snow_1h": float(data["snow"]),
            "clouds_all": int(data["clouds"]),
            "hour": int(data["hour"]),
            "day": int(data["day"]),
            "month": int(data["month"]),
            "year": int(data["year"])
        }])

        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[feature_columns]

        input_scaled = scaler.transform(input_df)

        input_scaled = np.repeat(
            input_scaled[:, np.newaxis, :],
            24,
            axis=1
        )

        prediction = model.predict(input_scaled)

        return jsonify({
            "prediction": float(prediction[0][0])
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
