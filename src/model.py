import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek

def plot_and_save_confusion_matrix(y_true, y_pred, model_name):
    """
    Plots and saves the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/confusion_matrix_{model_name.replace(' ', '_')}.png")
    plt.close()

def train_and_evaluate_model(
    model: RandomForestClassifier,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str
) -> Tuple[RandomForestClassifier, float, str]:
    """
    Trains and evaluates the model.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"{model_name} - Accuracy: {accuracy:.6f}")
    print(f"{model_name} - Classification Report:\n{report}")
    
    plot_and_save_confusion_matrix(y_test, y_pred, model_name)
    
    return model, accuracy, report

def train_model_classification(file_path: str) -> RandomForestClassifier:
    """
    Trains a Random Forest classification model using SMOTETomek for data balancing and cross-validation.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully, Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    if "Risk" not in df.columns:
        raise ValueError("Dataset must contain a 'Risk' column.")

    X = df.drop(columns=["Risk"])
    y = df["Risk"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracies = []

    for train_index, val_index in skf.split(X_train_raw, y_train):
        X_train_fold_raw = X_train_raw.iloc[train_index]
        X_val_fold_raw = X_train_raw.iloc[val_index]
        y_train_fold = y_train.iloc[train_index]
        y_val_fold = y_train.iloc[val_index]

        scaler_fold = StandardScaler()
        X_train_fold = scaler_fold.fit_transform(X_train_fold_raw)
        X_val_fold = scaler_fold.transform(X_val_fold_raw)

        smote_tomek = SMOTETomek(random_state=42)
        X_train_fold_resampled, y_train_fold_resampled = smote_tomek.fit_resample(
            X_train_fold, y_train_fold
        )

        rf_model = RandomForestClassifier(
            max_depth=10,
            max_features="sqrt",
            min_samples_leaf=2,
            min_samples_split=5,
            n_estimators=300,
            random_state=42
        )
        rf_model.fit(X_train_fold_resampled, y_train_fold_resampled)
        val_accuracy = rf_model.score(X_val_fold, y_val_fold)
        cv_accuracies.append(val_accuracy)

    print(f"Cross-validation Accuracy Scores: {cv_accuracies}")
    print(f"Mean Accuracy: {np.mean(cv_accuracies):.6f}, Std Dev: {np.std(cv_accuracies):.6f}")

    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train_raw)
    X_test_scaled = scaler_final.transform(X_test_raw)

    smote_tomek_final = SMOTETomek(random_state=42)
    X_train_resampled, y_train_resampled = smote_tomek_final.fit_resample(
        X_train_scaled, y_train
    )

    final_model = RandomForestClassifier(
        max_depth=10,
        max_features="sqrt",
        min_samples_leaf=2,
        min_samples_split=5,
        n_estimators=300,
        random_state=42
    )

    trained_model, accuracy, report = train_and_evaluate_model(
        final_model, 
        X_train_resampled, 
        X_test_scaled, 
        y_train_resampled, 
        y_test,
        "Random Forest Classifier"
    )

    os.makedirs("models", exist_ok=True)

    with open("models/risk_classification_model.pkl", "wb") as f:
        pickle.dump(trained_model, f)
    
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler_final, f)

    print("Model and scaler saved successfully.")
    print("Model training completed.")

    return trained_model