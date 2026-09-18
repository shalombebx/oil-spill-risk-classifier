import pandas as pd
import joblib
import os
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

def train_and_evaluate_models(data_dir='data/processed', save_dir='models'):
    """Trains baseline and optimized models, saving the best performer[cite: 1, 2]."""
    X_train = pd.read_csv(f'{data_dir}/X_train.csv')
    X_test = pd.read_csv(f'{data_dir}/X_test.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv').values.ravel()
    y_test = pd.read_csv(f'{data_dir}/y_test.csv').values.ravel()
    
    # Target Encoding
    target_le = LabelEncoder()
    y_train_num = target_le.fit_transform(y_train)
    y_test_num = target_le.transform(y_test)
    
    # Baseline
    dummy_clf = DummyClassifier(strategy='most_frequent')
    dummy_clf.fit(X_train, y_train_num)
    y_pred_dummy = dummy_clf.predict(X_test)
    print(f"Dummy Classifier Weighted F1: {f1_score(y_test_num, y_pred_dummy, average='weighted'):.4f}")
    
    # Random Forest
    rf_clf = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf_clf.fit(X_train, y_train_num)
    y_pred_rf = rf_clf.predict(X_test)
    print(f"Random Forest Weighted F1: {f1_score(y_test_num, y_pred_rf, average='weighted'):.4f}")
    
    # XGBoost Tuning
    xgb_base = XGBClassifier(random_state=42, eval_metric='mlogloss')
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(xgb_base, param_grid, scoring='f1_weighted', cv=cv, n_jobs=-1)
    grid_search.fit(X_train, y_train_num)
    
    best_xgb = grid_search.best_estimator_
    y_pred_xgb = best_xgb.predict(X_test)
    
    print(f"Optimized XGBoost Weighted F1: {f1_score(y_test_num, y_pred_xgb, average='weighted'):.4f}")
    print(classification_report(y_test_num, y_pred_xgb, target_names=target_le.classes_))
    
    # Serialization
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(best_xgb, f'{save_dir}/xgboost_model.pkl')
    joblib.dump(target_le, f'{save_dir}/target_encoder.pkl')
