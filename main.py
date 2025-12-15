import sys, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add src folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import project modules
from data_preprocessing import preprocess_data
from train_ml_models import train_ml_models
from train_ann import train_ann
from evaluate import evaluate_models
from explainability import shap_explain_ml

def main():
    # ------------------ 1. Preprocess Data ------------------
    X_train, X_test, y_train, y_test, X_train_pca, X_test_pca, pca = preprocess_data("data/data.csv")

    # ------------------ 2. Simple Visualizations ------------------
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)

    df_train = X_train.copy()
    df_train['diagnosis'] = y_train

    # Drop all NaN rows
    df_train = df_train.dropna().reset_index(drop=True)

    # Encode diagnosis as integer
    if df_train['diagnosis'].dtype == 'object':
        df_train['diagnosis'] = df_train['diagnosis'].map({'B':0,'M':1})
    df_train['diagnosis'] = df_train['diagnosis'].astype(int)

    # Convert to string for Seaborn palette
    df_train['diagnosis_str'] = df_train['diagnosis'].astype(str)

    os.makedirs("outputs/figures", exist_ok=True)

    # Class distribution
    plt.figure(figsize=(5,4))
    sns.countplot(x='diagnosis_str', data=df_train, palette={'0':"#2ecc71",'1':"#e74c3c"})
    plt.xlabel("Diagnosis")
    plt.title("Class Distribution")
    plt.savefig("outputs/figures/class_distribution.png")
    plt.close()

    # Correlation heatmap
    df_numeric = df_train.select_dtypes(include=['float64','int64'])
    plt.figure(figsize=(10,8))
    sns.heatmap(df_numeric.corr(), cmap='coolwarm', annot=False)
    plt.title("Correlation Heatmap")
    plt.savefig("outputs/figures/correlation_heatmap.png")
    plt.close()

    # ------------------ 3. Train ML Models ------------------
    predictions, models = train_ml_models(X_train, X_test, y_train, y_test)

    # ------------------ 4. Train ANN ------------------
    ann_predictions, history, ann_model = train_ann(X_train, X_test, y_train, y_test)

    # ------------------ 5. Evaluate Models ------------------
    results_df = evaluate_models(y_test, predictions, (ann_predictions, history, ann_model))

    # ------------------ 6. Feature Importance (for RF/XGB) ------------------
    raw_feature_names = pd.read_csv("data/data.csv").drop(['id','diagnosis'], axis=1).columns
    for model_name in ['Random Forest','XGBoost']:
        if model_name in models and hasattr(models[model_name],'feature_importances_'):
            importances = models[model_name].feature_importances_
            feature_names = raw_feature_names[:len(importances)]  # match lengths
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(10,6))
            plt.title(f"Feature Importance - {model_name}")
            plt.bar(range(len(feature_names)), importances[indices], color='skyblue', align='center')
            plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(f"outputs/figures/feature_importance_{model_name}.png")
            plt.close()

    print("\nAll visualizations and feature importance plots saved in 'outputs/figures/'")
    print("Tabulated model results saved in 'outputs/results.csv'")

if __name__ == "__main__":
    main()
