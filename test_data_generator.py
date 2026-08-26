#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test data generator for the Data Analysis System
Author: Stephen Bilodeau
Date: August 13, 2025
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def generate_test_dataset(output_path='test_data', n_samples=1000, seed=42):
    """Generate a test dataset for the Data Analysis System.

    Args:
        output_path: Directory to save the test data
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Generate time series data
    start_date = datetime(2025, 1, 1)
    dates = [start_date + timedelta(hours=i*6) for i in range(n_samples)]

    # Generate sensor readings with seasonal patterns and trends
    time_index = np.arange(n_samples)

    # Temperature with seasonality + trend + noise
    temp_trend = 0.01 * time_index  # Slight upward trend
    temp_seasonality = 5 * np.sin(2 * np.pi * time_index / (4 * 24))  # Daily cycle (4 readings per day)
    temp_noise = np.random.normal(0, 1, n_samples)
    temperature = 20 + temp_trend + temp_seasonality + temp_noise

    # Pressure with different seasonality + trend + noise
    pressure_trend = -0.005 * time_index  # Slight downward trend
    pressure_seasonality = 2 * np.sin(2 * np.pi * time_index / (4 * 24 * 7))  # Weekly cycle
    pressure_noise = np.random.normal(0, 0.5, n_samples)
    pressure = 1013 + pressure_trend + pressure_seasonality + pressure_noise

    # Humidity correlates with temperature
    humidity = 70 - 0.8 * (temperature - 20) + np.random.normal(0, 3, n_samples)

    # Wind speed as independent variable
    wind_speed = 5 + np.random.gamma(2, 1.5, n_samples)

    # Precipitation with many zeros (intermittent rain) and some outliers
    precipitation = np.zeros(n_samples)
    rain_days = np.random.choice(n_samples, size=int(n_samples * 0.2), replace=False)
    precipitation[rain_days] = np.random.exponential(2, size=len(rain_days))
    # Add a few heavy rain outliers
    heavy_rain = np.random.choice(rain_days, size=10, replace=False)
    precipitation[heavy_rain] = np.random.uniform(10, 20, size=10)

    # Solar radiation with daily pattern and cloud influence
    solar_base = np.maximum(0, np.sin(2 * np.pi * (time_index % (4 * 24)) / (4 * 24)))
    cloud_effect = np.ones(n_samples)
    cloud_days = np.random.choice(n_samples, size=int(n_samples * 0.3), replace=False)
    cloud_effect[cloud_days] = np.random.uniform(0.2, 0.7, size=len(cloud_days))
    solar_radiation = 1000 * solar_base * cloud_effect

    # Composite quality score based on other variables
    quality_score = (
        0.3 * (temperature - 10) / 20 +
        0.2 * (pressure - 1000) / 20 +
        0.15 * (60 - humidity) / 30 +
        0.1 * (10 - wind_speed) / 10 +
        0.15 * (1 - precipitation / 5) +
        0.1 * solar_radiation / 1000
    )
    # Normalize to 0-100 range
    quality_score = 100 * (quality_score - np.min(quality_score)) / (np.max(quality_score) - np.min(quality_score))

    # Create anomalies in a few samples
    anomaly_indices = np.random.choice(n_samples, size=20, replace=False)
    temperature[anomaly_indices] += np.random.uniform(10, 15, size=20)
    pressure[anomaly_indices] += np.random.uniform(-15, 15, size=20)
    humidity[anomaly_indices] = np.random.uniform(0, 100, size=20)

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'temperature': temperature,
        'pressure': pressure,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'precipitation': precipitation,
        'solar_radiation': solar_radiation,
        'quality_score': quality_score
    })

    # Save as CSV
    csv_path = os.path.join(output_path, 'sensor_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"CSV data saved to {csv_path}")

    # Save a subset as JSON
    json_data = df.head(100).to_dict(orient='records')
    json_path = os.path.join(output_path, 'sensor_data_sample.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON sample saved to {json_path}")

    # Create a visualization of the data
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 2, 1)
    plt.plot(df['timestamp'], df['temperature'])
    plt.title('Temperature Over Time')
    plt.xticks(rotation=45)

    plt.subplot(3, 2, 2)
    plt.plot(df['timestamp'], df['pressure'])
    plt.title('Pressure Over Time')
    plt.xticks(rotation=45)

    plt.subplot(3, 2, 3)
    plt.plot(df['timestamp'], df['humidity'])
    plt.title('Humidity Over Time')
    plt.xticks(rotation=45)

    plt.subplot(3, 2, 4)
    plt.plot(df['timestamp'], df['precipitation'])
    plt.title('Precipitation Over Time')
    plt.xticks(rotation=45)

    plt.subplot(3, 2, 5)
    plt.plot(df['timestamp'], df['solar_radiation'])
    plt.title('Solar Radiation Over Time')
    plt.xticks(rotation=45)

    plt.subplot(3, 2, 6)
    plt.plot(df['timestamp'], df['quality_score'])
    plt.title('Quality Score Over Time')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plot_path = os.path.join(output_path, 'sensor_data_overview.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Overview plot saved to {plot_path}")

    return csv_path, json_path

if __name__ == "__main__":
    csv_path, json_path = generate_test_dataset()
    print(f"Test dataset generated. Use with data_analysis_system.py:")
    print(f"python data_analysis_system.py --data {csv_path} --analysis all")
