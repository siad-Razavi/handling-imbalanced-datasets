# src/data_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

def load_data(filepath: str) -> pd.DataFrame:
    """Loads dataset from the specified filepath."""
    df = pd.read_csv(filepath)
    return df

def preprocess_and_split(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """Separates features and target, and splits into train and test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test
