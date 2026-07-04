import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Union

class FloodPredictor:
    """
    Class to manage prediction requests. Loads the saved model and scaler,
    validates input parameters, and runs predictions.
    """
    def __init__(self, model_path: str = "models/model.pkl", scaler_path: str = "models/scaler.pkl"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.feature_cols = [
            "Annual Rainfall",
            "Cloud Visibility",
            "Seasonal Rainfall",
            "Temperature",
            "Humidity",
            "Pressure",
            "River Level",
            "Wind Speed",
            "Monsoon Intensity",
            "Average Rainfall"
        ]
        self._load_resources()

    def _load_resources(self):
        """
        Loads the pre-trained machine learning model and standard scaler.
        """
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            raise FileNotFoundError(
                "Model files not found. Please train models first using model_training.py."
            )
            
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
            
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
            
        print("Model and Scaler loaded successfully.")

    def validate_inputs(self, inputs: Dict[str, Any]) -> Dict[str, float]:
        """
        Validates input weather and rainfall parameters.
        Raises ValueError if fields are missing or numbers are invalid.
        """
        validated = {}
        for col in self.feature_cols:
            if col not in inputs:
                raise ValueError(f"Missing required parameter: {col}")
            
            try:
                val = float(inputs[col])
            except (ValueError, TypeError):
                raise ValueError(f"Parameter {col} must be a number.")
                
            # Business logic validation for physical boundaries
            if col in ["Annual Rainfall", "Seasonal Rainfall", "Average Rainfall", "River Level", "Wind Speed"] and val < 0:
                raise ValueError(f"{col} cannot be negative.")
                
            if col == "Humidity" and not (0 <= val <= 100):
                raise ValueError("Humidity must be between 0% and 100%.")
                
            if col == "Cloud Visibility" and not (0 <= val <= 100):
                raise ValueError("Cloud Visibility must be between 0% and 100%.")
                
            if col == "Monsoon Intensity" and not (1 <= val <= 10):
                raise ValueError("Monsoon Intensity must be on a scale of 1 to 10.")
                
            if col == "Pressure" and not (900 <= val <= 1100):
                raise ValueError("Pressure must be between 900 hPa and 1100 hPa (realistic atmospheric bounds).")
                
            validated[col] = val
            
        return validated

    def predict(self, inputs: Dict[str, Any]) -> Tuple[int, float]:
        """
        Predicts flood likelihood for a single input record.
        Returns:
            Tuple[int, float]: (prediction_class (0 or 1), probability_percentage)
        """
        # 1. Validate inputs
        clean_inputs = self.validate_inputs(inputs)
        
        # 2. Convert to DataFrame to ensure correct column order
        df = pd.DataFrame([clean_inputs])[self.feature_cols]
        
        # 3. Scale inputs
        scaled_features = self.scaler.transform(df)
        
        # 4. Predict probability and class
        # XGBoost, RF, DecisionTree, KNN support predict_proba
        prob = self.model.predict_proba(scaled_features)[0][1]
        pred_class = int(self.model.predict(scaled_features)[0])
        
        return pred_class, float(prob)

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs batch prediction on a Pandas DataFrame.
        Cleans missing values by imputing column medians (using standard rules),
        scales features, makes predictions, and returns the DataFrame with prediction columns.
        """
        # Ensure all feature columns exist
        missing = [col for col in self.feature_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Uploaded CSV is missing required columns: {', '.join(missing)}")
            
        # Extract features and handle missing data by copying and fillna (using simple median or 0)
        df_features = df[self.feature_cols].copy()
        for col in self.feature_cols:
            median_val = df_features[col].median()
            # If the column is entirely empty/NaN, use a default fallback
            if pd.isna(median_val):
                median_val = 0.0
            df_features[col] = df_features[col].fillna(median_val)
            
        # Scale
        scaled_features = self.scaler.transform(df_features)
        
        # Predict
        probs = self.model.predict_proba(scaled_features)[:, 1]
        preds = self.model.predict(scaled_features)
        
        # Append predictions to the original dataframe
        output_df = df.copy()
        output_df["Flood Probability (%)"] = np.round(probs * 100, 2)
        output_df["Prediction"] = np.where(preds == 1, "Flood Expected", "No Flood Expected")
        
        return output_df
