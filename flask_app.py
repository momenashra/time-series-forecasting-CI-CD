# -*- coding: utf-8 -*-
"""
Flask API for Time-Series Forecasting
Created on Tue March 18, 2025
@author: Momen.Ashraf
"""

from flask import Flask, request, jsonify
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger

# Initialize Flask App
app = Flask(__name__)
Swagger(app)

# Load the trained forecasting model
with open("forecaster.pkl", "rb") as model_file:
    forecaster = pickle.load(model_file)

@app.route('/base')
def welcome():
    return "Welcome to Time-Series Forecasting API"

@app.route('/values', methods=["GET"])
def predict_forecast():
    """
    Predict Future Values in Time-Series
    ---
    parameters:
      - name: steps
        in: query
        type: integer
        required: true
        description: Number of future time steps to predict
    responses:
        200:
            description: Forecasted values
    """
    steps = request.args.get("steps", default=10, type=int)  # Default 10 steps
    try:
        prediction = forecaster.predict(steps)  # Assuming your model supports `.predict()`
        return jsonify({"forecast": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/', methods=["POST"])
def predict_from_file():
    """
    Predict Future Values from Uploaded Time-Series File
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: CSV file containing time-series data
    responses:
        200:
            description: Forecasted values
    """
    try:
        file = request.files['file']
        df = pd.read_csv(test.csv)
        
        # Ensure the correct input format
        if df.shape[1] != 1:
            return jsonify({"error": "Uploaded CSV must have exactly 1 column with time-series values"})

        data = df.values.reshape(-1, 1)
        prediction = forecaster.predict(len(data))  # Predict for the same length
        return jsonify({"forecast": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
