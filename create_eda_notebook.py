import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def save_eda_plots(data_path: str = "dataset/flood.csv", out_dir: str = "static/images"):
    """
    Generates and saves the EDA visualization charts to static/images/
    so they can be displayed dynamically on the Flask dashboard.
    """
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(data_path):
        print(f"Data path {data_path} not found. Skipping plot generation.")
        return
        
    df = pd.read_csv(data_path)
    
    # Fill NAs temporarily for plots
    df_imputed = df.fillna(df.median(numeric_only=True))
    
    # 1. Target Distribution count plot
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Flood Label", data=df_imputed, palette="Blues")
    plt.title("Distribution of Target: Flood Label", fontsize=12, fontweight="bold")
    plt.xlabel("Flood Label (0: No Flood, 1: Flood)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eda_target_dist.png"), dpi=150)
    plt.close()
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr = df_imputed.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Correlation Matrix Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eda_correlation.png"), dpi=150)
    plt.close()
    
    # 3. Distribution Plot: River Level and Seasonal Rainfall
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df_imputed["River Level"], kde=True, ax=axes[0], color="skyblue")
    axes[0].set_title("River Level Distribution", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("River Level (m)")
    
    sns.histplot(df_imputed["Seasonal Rainfall"], kde=True, ax=axes[1], color="salmon")
    axes[1].set_title("Seasonal Rainfall Distribution", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Rainfall (mm)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eda_distributions.png"), dpi=150)
    plt.close()
    
    # 4. Box Plots for Outlier Detection (River Level and Monsoon Intensity)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(y="River Level", x="Flood Label", data=df_imputed, ax=axes[0], palette="Blues")
    axes[0].set_title("River Level vs Flood Label (Outliers)", fontsize=11, fontweight="bold")
    
    sns.boxplot(y="Seasonal Rainfall", x="Flood Label", data=df_imputed, ax=axes[1], palette="Oranges")
    axes[1].set_title("Seasonal Rainfall vs Flood Label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "eda_boxplots.png"), dpi=150)
    plt.close()
    
    print(f"Saved EDA charts to {out_dir}")

def generate_ipynb(notebook_path: str = "notebooks/Flood_EDA.ipynb"):
    """
    Creates a Jupyter Notebook containing markdown descriptions and execution code for EDA.
    """
    os.makedirs(os.path.dirname(notebook_path), exist_ok=True)
    
    # We construct the notebook JSON dictionary structure
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Rising Waters: Exploratory Data Analysis (EDA)\n",
                    "### A Machine Learning Approach to Flood Prediction\n",
                    "\n",
                    "This notebook performs a comprehensive exploratory analysis on our flood prediction dataset. The dataset contains environmental, atmospheric, and rainfall parameters. Our target is to predict whether a flood is expected (`Flood Label` = 1) or not (`Flood Label` = 0)."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Import Libraries and Load Dataset"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "%matplotlib inline\n",
                    "\n",
                    "# Load the dataset\n",
                    "df = pd.read_csv('../dataset/flood.csv')\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Dataset Information\n",
                    "We check the number of rows, columns, data types, and check for missing and duplicate values."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Basic info\n",
                    "print(\"=== Dataset Info ===\")\n",
                    "df.info()\n",
                    "\n",
                    "# Check for missing values\n",
                    "print(\"\\n=== Missing Values ===\")\n",
                    "missing = df.isnull().sum()\n",
                    "print(missing[missing > 0])\n",
                    "\n",
                    "# Check for duplicates\n",
                    "print(f\"\\nDuplicate rows found: {df.duplicated().sum()}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Statistical Summary\n",
                    "Review descriptive statistics to understand centers, ranges, and variations."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df.describe()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Data Visualizations"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 4.1 Target Variable Count Plot\n",
                    "Checking the class distribution (balance) of the target variable `Flood Label`."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(6, 4))\n",
                    "sns.countplot(x='Flood Label', data=df, palette='Blues')\n",
                    "plt.title('Distribution of Flood Labels')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 4.2 Correlation Matrix Heatmap\n",
                    "Understand how features relate to one another and to the target label. High positive correlation indicates features that influence flood occurrences."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(10, 8))\n",
                    "sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', square=True)\n",
                    "plt.title('Correlation Heatmap')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 4.3 Feature Distribution Plots\n",
                    "We display histograms and density estimates of key predictors, like `River Level` and `Seasonal Rainfall`."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n",
                    "sns.histplot(df['River Level'], kde=True, ax=axes[0], color='skyblue')\n",
                    "axes[0].set_title('River Level Distribution')\n",
                    "\n",
                    "sns.histplot(df['Seasonal Rainfall'], kde=True, ax=axes[1], color='salmon')\n",
                    "axes[1].set_title('Seasonal Rainfall Distribution')\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 4.4 Outlier Detection Boxplots\n",
                    "Boxplots help detect extreme outliers in key features."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n",
                    "sns.boxplot(y='River Level', x='Flood Label', data=df, ax=axes[0], palette='Blues')\n",
                    "axes[0].set_title('River Level by Flood Target')\n",
                    "\n",
                    "sns.boxplot(y='Seasonal Rainfall', x='Flood Label', data=df, ax=axes[1], palette='Oranges')\n",
                    "axes[1].set_title('Seasonal Rainfall by Flood Target')\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 4.5 Top Feature Correlations\n",
                    "Let's rank features by their absolute correlation with the target `Flood Label`."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "correlations = df.corr()['Flood Label'].sort_values(ascending=False)\n",
                    "print(\"Correlation with Flood Label:\")\n",
                    "print(correlations)"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(notebook_path, "w") as f:
        json.dump(notebook, f, indent=2)
    print(f"Generated Jupyter Notebook at {notebook_path}")

if __name__ == "__main__":
    generate_ipynb()
    save_eda_plots()
