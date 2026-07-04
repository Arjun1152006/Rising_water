import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Tuple

DB_PATH = "predictions.db"

def get_db_connection() -> sqlite3.Connection:
    """
    Establishes and returns a connection to the SQLite database.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initializes the database and creates the predictions table if it does not exist.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            annual_rainfall REAL,
            cloud_visibility REAL,
            seasonal_rainfall REAL,
            temperature REAL,
            humidity REAL,
            pressure REAL,
            river_level REAL,
            wind_speed REAL,
            monsoon_intensity REAL,
            average_rainfall REAL,
            prediction_label TEXT NOT NULL,
            probability REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print("Database initialized successfully.")

def add_prediction(
    inputs: Dict[str, float], 
    prediction_label: str, 
    probability: float
) -> int:
    """
    Inserts a new prediction record into the database.
    Returns the ID of the newly inserted row.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO predictions (
            timestamp, annual_rainfall, cloud_visibility, seasonal_rainfall, 
            temperature, humidity, pressure, river_level, wind_speed, 
            monsoon_intensity, average_rainfall, prediction_label, probability
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp,
        inputs.get("Annual Rainfall"),
        inputs.get("Cloud Visibility"),
        inputs.get("Seasonal Rainfall"),
        inputs.get("Temperature"),
        inputs.get("Humidity"),
        inputs.get("Pressure"),
        inputs.get("River Level"),
        inputs.get("Wind Speed"),
        inputs.get("Monsoon Intensity"),
        inputs.get("Average Rainfall"),
        prediction_label,
        probability
    ))
    
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id

def get_all_predictions() -> List[Dict[str, Any]]:
    """
    Retrieves all predictions from the database, ordered by timestamp descending.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    
    predictions = []
    for row in rows:
        predictions.append(dict(row))
    return predictions

def delete_prediction(prediction_id: int) -> bool:
    """
    Deletes a specific prediction by ID. Returns True if a row was deleted.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
    conn.commit()
    deleted_rows = cursor.rowcount
    conn.close()
    return deleted_rows > 0

def clear_history() -> int:
    """
    Deletes all prediction history. Returns the number of deleted rows.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    deleted_rows = cursor.rowcount
    conn.close()
    return deleted_rows

if __name__ == "__main__":
    init_db()
