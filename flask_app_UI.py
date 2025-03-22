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
import os

# Initialize Flask App
app = Flask(__name__)
Swagger(app)

# Define forecaster as a global variable
forecaster = None

# Load the trained forecasting model
model_path = "forecaster.pkl"

if not os.path.exists(model_path):
    print(f"❌ Model file {model_path} NOT found in container!")
else:
    try:
        with open(model_path, "rb") as f:
            forecaster = pickle.load(f)
            print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        forecaster = None

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
    # Use the global forecaster variable
    global forecaster

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
