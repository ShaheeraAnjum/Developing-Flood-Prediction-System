import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    
    # Generate 5 independent features
    annual_rainfall = np.random.normal(1200, 400, num_samples) # in mm
    cloud_visibility = np.random.normal(5, 2, num_samples) # lower means more clouds
    seasonal_rainfall = np.random.normal(800, 300, num_samples)
    temperature = np.random.normal(28, 5, num_samples)
    humidity = np.random.normal(75, 10, num_samples)
    
    # Create the target variable based on some logic
    flood_probability = (
        0.4 * (annual_rainfall / 2000) +
        0.3 * (seasonal_rainfall / 1500) +
        0.2 * (humidity / 100) -
        0.1 * (cloud_visibility / 10)
    )
    
    threshold = np.percentile(flood_probability, 65) # Top 35% are floods
    classes = (flood_probability >= threshold).astype(int) # 1 for Flood, 0 for No Flood
    
    # Introduce outliers and missing values
    annual_rainfall[np.random.choice(num_samples, 20)] = 5000 
    temperature[np.random.choice(num_samples, 10)] = np.nan 
    
    df = pd.DataFrame({
        'Annual Rainfall': annual_rainfall,
        'Cloud Visibility': cloud_visibility,
        'Seasonal Rainfall': seasonal_rainfall,
        'Temperature': temperature,
        'Humidity': humidity,
        'class': classes
    })
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/flood_dataset.csv', index=False)
    print("Synthetic dataset generated at data/flood_dataset.csv")

if __name__ == "__main__":
    generate_synthetic_data()
