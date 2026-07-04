import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any, List

class DataPreprocessor:
    """
    A class to handle data preprocessing, including missing value imputation,
    duplicate removal, outlier treatment, and feature scaling.
    """
    def __init__(self, target_column: str = "Flood Label"):
        self.target_column = target_column
        self.scaler = StandardScaler()
        self.feature_cols: List[str] = []
        self.impute_values: Dict[str, float] = {}
        self.outlier_bounds: Dict[str, Tuple[float, float]] = {}

    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fits the preprocessor to the training data. Computes imputation medians
        and outlier boundaries based on the training set.
        """
        # Exclude target from features
        feature_data = df.drop(columns=[self.target_column], errors='ignore')
        self.feature_cols = list(feature_data.columns)

        for col in self.feature_cols:
            # 1. Store median for imputation
            self.impute_values[col] = float(feature_data[col].median())
            
            # 2. Store IQR bounds for outlier treatment
            q1 = feature_data[col].quantile(0.25)
            q3 = feature_data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            self.outlier_bounds[col] = (lower_bound, upper_bound)
            
        # Fit scaler on imputed, outlier-treated features
        processed_features = self._transform_features(feature_data)
        self.scaler.fit(processed_features)
        
        return self

    def _transform_features(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Internal method to clean features (impute missing, cap outliers).
        """
        df_clean = df_features.copy()
        
        for col in self.feature_cols:
            # Impute missing values
            if col in self.impute_values:
                df_clean[col] = df_clean[col].fillna(self.impute_values[col])
                
            # Outlier treatment (capping)
            if col in self.outlier_bounds:
                lower, upper = self.outlier_bounds[col]
                df_clean[col] = np.clip(df_clean[col], lower, upper)
                
        return df_clean

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms data by cleaning features and scaling them.
        Returns scaled feature matrix X and target array y.
        """
        # Separate features and target
        if self.target_column in df.columns:
            y = df[self.target_column].fillna(0).astype(int).values
            df_features = df.drop(columns=[self.target_column])
        else:
            y = np.zeros(len(df))
            df_features = df.copy()

        # Reorder columns to match fitted columns
        df_features = df_features[self.feature_cols]
        
        # Clean features
        cleaned_features = self._transform_features(df_features)
        
        # Scale features
        X_scaled = self.scaler.transform(cleaned_features)
        
        return X_scaled, y

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits and transforms in a single step.
        """
        return self.fit(df).transform(df)

    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans duplicates from the dataframe.
        """
        initial_rows = len(df)
        df_clean = df.drop_duplicates()
        dropped = initial_rows - len(df_clean)
        if dropped > 0:
            print(f"Removed {dropped} duplicate row(s).")
        return df_clean

    def save_scaler(self, path: str = "models/scaler.pkl"):
        """
        Saves the fitted StandardScaler to a file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)
        print(f"Saved scaler to {path}")

def load_scaler(path: str = "models/scaler.pkl") -> StandardScaler:
    """
    Loads a saved StandardScaler.
    """
    with open(path, "rb") as f:
        return pickle.load(f)

def run_preprocessing_pipeline(data_path: str = "dataset/flood.csv") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Runs the entire preprocessing pipeline: loading data, removing duplicates,
    fitting preprocessor, scaling, train-test split, and saving the scaler.
    """
    print("Starting preprocessing pipeline...")
    # Load dataset
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run generate_data.py first.")
        
    df = pd.read_csv(data_path)
    
    # Preprocessor initialization
    preprocessor = DataPreprocessor(target_column="Flood Label")
    
    # 1. Clean duplicates
    df_clean = preprocessor.clean_dataset(df)
    
    # 2. Train-Test Split (80% / 20%) before fitting to prevent data leakage
    train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=42, stratify=df_clean["Flood Label"])
    
    # 3. Fit preprocessor on training data, transform both train and test
    preprocessor.fit(train_df)
    
    X_train, y_train = preprocessor.transform(train_df)
    X_test, y_test = preprocessor.transform(test_df)
    
    # 4. Save scaler
    preprocessor.save_scaler("models/scaler.pkl")
    
    print("Preprocessing completed successfully.")
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, preprocessor.feature_cols

if __name__ == "__main__":
    run_preprocessing_pipeline()
