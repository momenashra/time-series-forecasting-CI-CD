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
from flasgger import Swagger, swag_from

# Initialize Flask App
app = Flask(__name__)
Swagger(app)

# Load the trained forecasting model
try:
    with open("forecaster.pkl", "rb") as model_file:
        forecaster = pickle.load(model_file)
except Exception as e:
    print(f"Error loading model: {e}")
    forecaster = None  # Prevent errors if the model fails to load

@app.route('/h')
@swag_from({
    'responses': {
        200: {
            'description': 'Welcome message',
            'examples': {'text': 'Welcome to Time-Series Forecasting API'}
        }
    }
})
def welcome():
    return "Welcome to Time-Series Forecasting API"

@app.route('/g', methods=["GET"])
@swag_from({
    'parameters': [
        {
            'name': 'forecast_horizon',
            'in': 'query',
            'type': 'integer',
            'default': 10,
            'required': False,
            'description': 'Number of steps to forecast (default: 10)'
        }
    ],
    'responses': {
        200: {
            'description': 'Predicted future values',
            'schema': {
                'type': 'object',
                'properties': {
                    'forecast': {'type': 'array', 'items': {'type': 'number'}}
                }
            }
        }
    }
})
def predict_forecast():
    """
    Predict Future Values in Time-Series
    """
    if forecaster is None:
        return jsonify({"error": "Model not loaded. Train or load the model first."})

    try:
        steps = request.args.get("forecast_horizon", default=10, type=int)  # Default to 10 steps
        
        # Ensure the input matches the training shape (1 sample, 720 features)
        steps_array = np.random(1, 720)  # Create a dummy input with 720 features
        prediction = forecaster.predict(steps_array)  # Pass correctly shaped input
        
        return jsonify({"forecast": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/', methods=["POST"])
@swag_from({
    'parameters': [
        {
            'name': 'file',
            'in': 'formData',
            'type': 'file',
            'required': True,
            'description': 'CSV file containing time-series data'
        }
    ],
    'responses': {
        200: {
            'description': 'Predicted future values from file',
            'schema': {
                'type': 'object',
                'properties': {
                    'forecast': {'type': 'array', 'items': {'type': 'number'}}
                }
            }
        }
    }
})
def predict_from_file():
    """
    Predict Future Values from Uploaded Time-Series File
    """
    if forecaster is None:
        return jsonify({"error": "Model not loaded. Train or load the model first."})

    try:
        file = request.files['file']
        df = pd.read_csv(file)

        data = df.values.flatten()  # Convert DataFrame to 1D NumPy array

        # Ensure we have enough data points
        if len(data) < 720:
            return jsonify({"error": f"Insufficient data. Expected at least 720 values, got {len(data)}"})

        # Reshape data to match model input
        data = data[:720].reshape(1, -1)  # Use only first 720 points
        
        prediction = forecaster.predict(data)  # Pass correctly shaped input
        return jsonify({"forecast": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
