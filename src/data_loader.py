import pandas as pd
from pathlib import Path
from typing import Optional

def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Loads a CSV file into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        Optional[pd.DataFrame]: Loaded DataFrame if successful, otherwise None.
    """
    file = Path(file_path)
    
    if not file.exists():
        print(f"Error: File '{file_path}' not found.")
        return None
    
    try:
        df = pd.read_csv(file)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None