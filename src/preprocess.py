"""
preprocess.py
=============
Data cleaning, encoding, and feature engineering utilities.
Refactored from 02_preprocessing.ipynb — Phase 4.

Usage:
    from src.preprocess import load_raw, normalise_target, fill_missing, encode_features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# ── Column mapping (update left side to match your CSV) ──────────────────────
COLUMN_MAP = {
    'Date':                     'date',
    'State':                    'state',
    'Local Government Area':    'lga',
    'Company':                  'company',
    'Cause':                    'cause',
    'Quantity Spilled (bbls)':  'quantity',
    'Category':                 'category',   # target
    'Spill Location':           'environment',
    'JIV':                      'jiv',
}

TARGET   = 'category'
FEATURES = ['cause', 'company', 'state', 'lga', 'environment', 'jiv']


# ── Loading ───────────────────────────────────────────────────────────────────

def load_raw(path: str) -> pd.DataFrame:
    """Load the raw NOSDRA CSV from disk."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows | {df.shape[1]} columns")
    return df


def rename_columns(df: pd.DataFrame, column_map: dict = None) -> pd.DataFrame:
    """Rename columns using COLUMN_MAP; skip keys not in the DataFrame."""
    col_map = column_map or COLUMN_MAP
    col_map = {k: v for k, v in col_map.items() if k in df.columns}
    return df.rename(columns=col_map).copy()


# ── Cleaning ──────────────────────────────────────────────────────────────────

def normalise_target(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Standardise target column casing: 'MINOR' -> 'Minor', 'major' -> 'Major'."""
    df = df.copy()
    df[target] = df[target].astype(str).str.strip().str.title()
    return df


def extract_time_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """Parse date column and extract year and month features."""
    df = df.copy()
    df['date_parsed'] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
    df['year']        = df['date_parsed'].dt.year
    df['month']       = df['date_parsed'].dt.month
    return df


def fill_missing(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Impute missing values:
      - Categorical columns: fill with 'Unknown'
      - Numeric columns: fill with column median
    """
    df = df.copy()
    for col in features:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('Unknown')
            else:
                df[col] = df[col].fillna(df[col].median())
    return df


def drop_missing_target(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Drop rows where the target variable is missing."""
    n_before = len(df)
    df = df.dropna(subset=[target]).copy()
    print(f"Dropped {n_before - len(df)} rows with missing target | {len(df):,} remain")
    return df


# ── Encoding ──────────────────────────────────────────────────────────────────

def encode_features(df: pd.DataFrame, features: list) -> tuple:
    """
    Label-encode all categorical features in-place.

    Returns:
        df (pd.DataFrame): DataFrame with encoded features.
        encoders (dict): {column_name: LabelEncoder} for later inverse_transform.
    """
    df = df.copy()
    encoders = {}
    for col in features:
        if col in df.columns and df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            print(f"  Encoded '{col}' -> {len(le.classes_)} unique values")
    return df, encoders


def encode_target(df: pd.DataFrame, target: str = TARGET) -> tuple:
    """
    Label-encode the target variable.

    Returns:
        df (pd.DataFrame): DataFrame with encoded target.
        le_target (LabelEncoder): Encoder for inverse_transform.
    """
    df = df.copy()
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])
    print(f"Target classes: {list(enumerate(le.classes_))}")
    return df, le


# ── Pipeline ──────────────────────────────────────────────────────────────────

def build_model_dataset(df: pd.DataFrame, features: list, target: str = TARGET) -> tuple:
    """
    Full preprocessing pipeline: clean -> encode -> return X, y.

    Returns:
        X (pd.DataFrame): Encoded feature matrix.
        y (pd.Series): Encoded target vector.
        encoders (dict): Feature LabelEncoders.
        le_target (LabelEncoder): Target LabelEncoder.
    """
    all_feats = [f for f in features if f in df.columns]

    df = drop_missing_target(df, target)
    df = fill_missing(df, all_feats)
    df, encoders  = encode_features(df, all_feats)
    df, le_target = encode_target(df, target)

    X = df[all_feats]
    y = df[target]
    return X, y, encoders, le_target
