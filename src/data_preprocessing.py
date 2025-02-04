from typing import Tuple, Dict
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Preprocesses the dataset:
    - Drops unnecessary columns
    - Handles missing values
    - Encodes categorical variables
    - Encodes the 'Risk' column as 0 and 1

    Args:
        df (pd.DataFrame): Raw dataset.

    Returns:
        Tuple[pd.DataFrame, Dict[str, LabelEncoder]]: Processed DataFrame and encoders for categorical features.
    """

    # Drop unnecessary columns safely
    df = df.drop(columns=["Unnamed: 0", "Age"], errors="ignore")

    # Handle missing values in categorical columns
    df["Saving accounts"].fillna("unknown", inplace=True)
    df["Checking account"].fillna("unknown", inplace=True)

    # Extract target column before modifying the dataset
    if "Risk" not in df.columns:
        raise ValueError("Dataset must contain a 'Risk' column.")

    y = df["Risk"].map({"good": 1, "bad": 0})  # Encode 'Risk' as binary
    df = df.drop(columns=["Risk", "Credit amount"], errors="ignore")

    # Encode categorical variables
    categorical_cols = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
    label_encoders = {}

    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le  # Store encoder for future use

    # Add the encoded 'Risk' column back
    df["Risk"] = y

    return df, label_encoders
