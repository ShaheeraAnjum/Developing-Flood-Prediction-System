from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = joblib.load('model/floods.save')
    scaler = joblib.load('model/transform.save')
except Exception as e:
    print(f"Error loading model or scaler. Ensure you have run train.py. Details: {e}")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract features from form
            annual_rainfall = float(request.form['annual_rainfall'])
            cloud_visibility = float(request.form['cloud_visibility'])
            seasonal_rainfall = float(request.form['seasonal_rainfall'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            
            # Structure into DataFrame matching training data
            input_data = pd.DataFrame({
                'Annual Rainfall': [annual_rainfall],
                'Cloud Visibility': [cloud_visibility],
                'Seasonal Rainfall': [seasonal_rainfall],
                'Temperature': [temperature],
                'Humidity': [humidity]
            })
            
            # Scale the input
            scaled_input = scaler.transform(input_data)
            
            # Generate prediction
            prediction = model.predict(scaled_input)[0]
            
            # Redirect based on result
            if prediction == 1:
                return redirect(url_for('chance'))
            else:
                return redirect(url_for('no_chance'))
                
        except Exception as e:
            return f"An error occurred during prediction: {e}"
            
    return render_template('index.html')

@app.route('/chance')
def chance():
    return render_template('chance.html')

@app.route('/no_chance')
def no_chance():
    return render_template('no_chance.html')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
