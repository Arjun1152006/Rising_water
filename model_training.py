import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server-side generation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import xgboost as xgb
from typing import Dict, Any, Tuple

from preprocess import run_preprocessing_pipeline

class ModelTrainer:
    """
    Class to orchestrate training, evaluation, comparison, and serialization
    of different classification models.
    """
    def __init__(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, feature_names: list):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.best_model_name: str = ""
        self.best_model: Any = None

    def train_decision_tree(self) -> Any:
        """
        Trains and tunes a Decision Tree model.
        """
        print("\n--- Training Decision Tree ---")
        dt = DecisionTreeClassifier(random_state=42)
        param_grid = {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy']
        }
        grid = GridSearchCV(dt, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid.fit(self.X_train, self.y_train)
        print(f"DT Best Params: {grid.best_params_}")
        return grid.best_estimator_

    def train_random_forest(self) -> Any:
        """
        Trains and tunes a Random Forest model.
        """
        print("\n--- Training Random Forest ---")
        rf = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5]
        }
        grid = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid.fit(self.X_train, self.y_train)
        print(f"RF Best Params: {grid.best_params_}")
        return grid.best_estimator_

    def train_knn(self) -> Any:
        """
        Trains and tunes a K-Nearest Neighbors model.
        """
        print("\n--- Training K-Nearest Neighbors ---")
        knn = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance']
        }
        grid = GridSearchCV(knn, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid.fit(self.X_train, self.y_train)
        print(f"KNN Best Params: {grid.best_params_}")
        return grid.best_estimator_

    def train_xgboost(self) -> Any:
        """
        Trains and tunes an XGBoost model.
        """
        print("\n--- Training XGBoost ---")
        # XGBoost requires labels starting at 0, which they do (0 and 1)
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        grid = GridSearchCV(xgb_model, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid.fit(self.X_train, self.y_train)
        print(f"XGBoost Best Params: {grid.best_params_}")
        return grid.best_estimator_

    def evaluate_model(self, name: str, model: Any) -> Dict[str, Any]:
        """
        Evaluates a single model and compiles performance metrics.
        """
        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test)[:, 1]

        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred, zero_division=0)
        rec = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        
        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        auc_score = auc(fpr, tpr)
        cm = confusion_matrix(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, output_dict=True)

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "auc": auc_score,
            "cm": cm,
            "report": report,
            "fpr": fpr,
            "tpr": tpr,
            "estimator": model
        }

    def train_and_evaluate_all(self):
        """
        Runs the full model training and evaluation loop.
        """
        self.models['Decision Tree'] = self.train_decision_tree()
        self.models['Random Forest'] = self.train_random_forest()
        self.models['KNN'] = self.train_knn()
        self.models['XGBoost'] = self.train_xgboost()

        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            self.results[name] = self.evaluate_model(name, model)

    def print_comparison_table(self) -> pd.DataFrame:
        """
        Outputs a model comparison table.
        """
        comparison_data = []
        for name, res in self.results.items():
            comparison_data.append({
                "Model": name,
                "Accuracy": f"{res['accuracy'] * 100:.2f}%",
                "Precision": f"{res['precision'] * 100:.2f}%",
                "Recall": f"{res['recall'] * 100:.2f}%",
                "F1 Score": f"{res['f1_score'] * 100:.2f}%",
                "AUC Score": f"{res['auc']:.4f}"
            })
        df = pd.DataFrame(comparison_data)
        print("\n" + "="*50)
        print("MODEL COMPARISON TABLE")
        print("="*50)
        print(df.to_string(index=False))
        print("="*50)
        return df

    def select_and_save_best_model(self, models_dir: str = "models"):
        """
        Selects the model with the highest F1 Score and saves it to a file.
        Also creates the evaluation visualization plots.
        """
        best_f1 = -1.0
        for name, res in self.results.items():
            # Using F1 Score as the selection metric for flood safety classification
            if res["f1_score"] > best_f1:
                best_f1 = res["f1_score"]
                self.best_model_name = name
                self.best_model = res["estimator"]

        print(f"\nBest Model Selected: {self.best_model_name} with F1-Score: {best_f1*100:.2f}%")
        
        # Save model
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.best_model, f)
        print(f"Best model serialized and saved to {model_path}")
        
        # Save validation results to database or file for display
        results_summary = {
            "best_model_name": self.best_model_name,
            "metrics": {
                "accuracy": self.results[self.best_model_name]["accuracy"],
                "precision": self.results[self.best_model_name]["precision"],
                "recall": self.results[self.best_model_name]["recall"],
                "f1_score": self.results[self.best_model_name]["f1_score"],
                "auc": self.results[self.best_model_name]["auc"]
            }
        }
        with open(os.path.join(models_dir, "results_summary.pkl"), "wb") as f:
            pickle.dump(results_summary, f)

    def generate_and_save_plots(self, plots_dir: str = "static/images"):
        """
        Generates and saves the final ROC Curve and Confusion Matrix plots
        for comparison to static/images/.
        """
        os.makedirs(plots_dir, exist_ok=True)

        # Plot 1: ROC Curve Comparison
        plt.figure(figsize=(8, 6))
        for name, res in self.results.items():
            plt.plot(res["fpr"], res["tpr"], label=f"{name} (AUC = {res['auc']:.3f})", lw=2)
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "roc_curve_comparison.png"), dpi=150)
        plt.close()

        # Plot 2: Confusion Matrix for Best Model
        best_cm = self.results[self.best_model_name]["cm"]
        plt.figure(figsize=(6, 5))
        sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['No Flood Expected', 'Flood Expected'],
                    yticklabels=['No Flood Expected', 'Flood Expected'],
                    annot_kws={"size": 14, "weight": "bold"})
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(f'Confusion Matrix - {self.best_model_name} (Best Model)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "confusion_matrix_best.png"), dpi=150)
        plt.close()
        print(f"Generated validation charts in {plots_dir}")

def main():
    # 1. Preprocess data
    X_train, X_test, y_train, y_test, feature_names = run_preprocessing_pipeline()

    # 2. Train models
    trainer = ModelTrainer(X_train, X_test, y_train, y_test, feature_names)
    trainer.train_and_evaluate_all()

    # 3. Print table and save best model
    trainer.print_comparison_table()
    trainer.select_and_save_best_model()
    
    # 4. Export visualization charts
    trainer.generate_and_save_plots()

if __name__ == "__main__":
    main()
