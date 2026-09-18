import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

def categorize_severity(row):
    """Re-engineers the target variable based on volume and habitat[cite: 4]."""
    qty = row['estimatedquantity']
    habitat = str(row.get('spillareahabitat', '')).lower()
    
    if pd.isna(qty): return 'Unknown'
    if 'inland' in habitat:
        if qty > 250: return 'Major'
        elif qty >= 25: return 'Medium'
        else: return 'Minor'
    else:
        if qty > 2500: return 'Major'
        elif qty >= 250: return 'Medium'
        else: return 'Minor'

def preprocess_data(file_path):
    """Executes cleaning, imputation, encoding, and stratified splitting[cite: 4]."""
    df = pd.read_csv(file_path, on_bad_lines='skip')
    df['estimatedquantity'] = pd.to_numeric(df['estimatedquantity'], errors='coerce')
    
    # 1. Target Engineering
    df['severity'] = df.apply(categorize_severity, axis=1)
    df_clean = df[df['severity'] != 'Unknown'].copy()
    
    # 2. Drop Excluded Columns
    cols_to_drop = [
        'id', 'incidentnumber', 'descriptionofimpact', 'attachments', 
        'updatefor', 'zonaloffice', 'reportdate', 'estimatedquantity', 
        'quantityrecovered', 'spillstopdate', 'typeoffacility', 
        'initialcontainmentmeasures', 'latitude', 'longitude', 'lga', 
        'estimatedspillarea', 'formadate', 'formbdate', 'formcdate', 
        'jivdate', 'jivpresent', 'cleanupdate', 'cleanupcompleteddate', 
        'cleanupmethods', 'postcleanupinspectiondate', 'postimpactassessmentdate', 
        'remediationstart', 'remediationend', 'remediationtype', 'finalsamplingdate', 
        'finallabresultsdate', 'certificatedate', 'certificatenumber', 'lastupdatedby'
    ]
    df_clean.drop(columns=df_clean.columns.intersection(cols_to_drop), inplace=True)
    
    # 3. Feature Engineering
    df_clean['incidentdate'] = pd.to_datetime(df_clean['incidentdate'], errors='coerce')
    df_clean['incident_year'] = df_clean['incidentdate'].dt.year
    df_clean['incident_month'] = df_clean['incidentdate'].dt.month
    df_clean.drop(columns=['incidentdate'], inplace=True)
    
    # 4. Missing Value Imputation
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    df_clean[cat_cols] = df_clean[cat_cols].fillna('unknown')
    
    num_cols = df_clean.select_dtypes(include=['number']).columns
    for col in num_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
    # 5. Label Encoding
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        encoders[col] = le
        
    os.makedirs('models', exist_ok=True)
    joblib.dump(encoders, 'models/label_encoders.pkl')
    
    # 6. Data Splitting
    X = df_clean.drop(columns=['severity'])
    y = df_clean['severity']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    os.makedirs('data/processed', exist_ok=True)
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    return X_train, X_test, y_train, y_test
