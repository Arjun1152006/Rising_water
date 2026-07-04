import os
import numpy as np
import pandas as pd

def create_project_structure(base_dir: str = "."):
    """
    Creates the required directory structure for the Flood Prediction project.
    """
    directories = [
        "dataset",
        "models",
        "static/css",
        "static/js",
        "static/images",
        "templates",
        "notebooks",
        "screenshots"
    ]
    for directory in directories:
        path = os.path.join(base_dir, directory)
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")

def generate_flood_dataset(file_path: str, num_samples: int = 5000, seed: int = 42):
    """
    Generates a realistic synthetic dataset for flood prediction and saves it to a CSV file.
    """
    np.random.seed(seed)

    # 1. Feature Generation
    # Temperature in Celsius (15 to 45)
    temperature = np.random.uniform(15.0, 45.0, num_samples)
    
    # Humidity in percentage (30 to 100)
    humidity = np.random.uniform(30.0, 100.0, num_samples)
    
    # Atmospheric pressure in hPa (980 to 1020)
    pressure = np.random.uniform(980.0, 1020.0, num_samples)
    
    # Wind speed in km/h (0 to 100)
    wind_speed = np.random.uniform(0.0, 100.0, num_samples)
    
    # Cloud Visibility percentage (10 to 100) - higher humidity generally means lower visibility (higher cloud cover, i.e., lower visibility)
    # We add noise to make it realistic
    cloud_visibility = np.clip(110.0 - humidity * 0.8 + np.random.normal(0, 10, num_samples), 10.0, 100.0)
    
    # Monsoon Intensity on a scale of 1 to 10
    monsoon_intensity = np.random.uniform(1.0, 10.0, num_samples)
    
    # Seasonal Rainfall in mm (200 to 2000)
    # Strongly dependent on monsoon intensity
    seasonal_rainfall = np.clip(
        monsoon_intensity * 150.0 + np.random.normal(300, 150, num_samples),
        200.0, 2000.0
    )
    
    # Average Rainfall in mm (100 to 1000)
    average_rainfall = np.random.uniform(100.0, 1000.0, num_samples)
    
    # Annual Rainfall in mm (1000 to 5000)
    # Related to seasonal rainfall and average rainfall
    annual_rainfall = np.clip(
        seasonal_rainfall * 2.0 + average_rainfall + np.random.normal(500, 300, num_samples),
        1000.0, 5000.0
    )
    
    # River Level in meters (1.0 to 15.0)
    # Highly correlated with seasonal rainfall and monsoon intensity
    river_level = np.clip(
        1.0 + (seasonal_rainfall / 200.0) * 1.2 + (monsoon_intensity * 0.4) + np.random.normal(0, 0.8, num_samples),
        1.0, 15.0
    )
    
    # 2. Target Generation (Flood Label: 0 or 1)
    # We define a log-odds relationship based on physical factors
    # Floods are highly likely when river levels are high, monsoon intensity is high, and humidity/rainfall are high.
    # Standardize values for the logistic function
    norm_river = (river_level - 5.0) / 3.0
    norm_seasonal = (seasonal_rainfall - 1000.0) / 400.0
    norm_monsoon = (monsoon_intensity - 5.0) / 2.5
    norm_humidity = (humidity - 70.0) / 15.0
    
    # Calculate log-odds
    log_odds = (
        2.5 * norm_river + 
        1.5 * norm_seasonal + 
        1.2 * norm_monsoon + 
        0.5 * norm_humidity - 
        0.8  # bias to balance classes
    )
    
    # Probability of flood
    probability = 1.0 / (1.0 + np.exp(-log_odds))
    
    # Generate binary labels (0 = No Flood, 1 = Flood)
    flood_label = np.where(probability >= 0.5, 1, 0)
    
    # Introduce some stochastic noise: flip 3% of labels randomly to simulate real-world noise
    flip_mask = np.random.rand(num_samples) < 0.03
    flood_label[flip_mask] = 1 - flood_label[flip_mask]
    
    # Create DataFrame
    df = pd.DataFrame({
        "Annual Rainfall": np.round(annual_rainfall, 1),
        "Cloud Visibility": np.round(cloud_visibility, 1),
        "Seasonal Rainfall": np.round(seasonal_rainfall, 1),
        "Temperature": np.round(temperature, 1),
        "Humidity": np.round(humidity, 1),
        "Pressure": np.round(pressure, 1),
        "River Level": np.round(river_level, 2),
        "Wind Speed": np.round(wind_speed, 1),
        "Monsoon Intensity": np.round(monsoon_intensity, 1),
        "Average Rainfall": np.round(average_rainfall, 1),
        "Flood Label": flood_label
    })
    
    # Introduce a small percentage of missing values (0.5%) to demonstrate missing value handling in preprocess.py
    for col in df.columns:
        if col != "Flood Label":
            mask = np.random.rand(num_samples) < 0.005
            df.loc[mask, col] = np.nan
            
    # Introduce duplicate rows (10 duplicate rows) to test duplicate removal
    duplicates = df.sample(n=10, random_state=42)
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Save to file
    df.to_csv(file_path, index=False)
    print(f"Generated flood dataset with {df.shape[0]} rows and {df.shape[1]} columns at {file_path}")
    print(f"Target variable distributions:\n{df['Flood Label'].value_counts(normalize=True)}")

if __name__ == "__main__":
    create_project_structure()
    generate_flood_dataset("dataset/flood.csv")
