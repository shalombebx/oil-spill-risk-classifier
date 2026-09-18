import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np

def generate_global_explanations(data_dir='data/processed', model_dir='models'):
    """Generates global SHAP summary and beeswarm plots[cite: 2]."""
    X_test = pd.read_csv(f'{data_dir}/X_test.csv')
    model = joblib.load(f'{model_dir}/xgboost_model.pkl') 
    target_le = joblib.load(f'{model_dir}/target_encoder.pkl')
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Feature Importance Bar Chart
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", class_names=target_le.classes_, show=False)
    plt.tight_layout()
    plt.savefig('reports/figures/shap_global_bar.png', dpi=300)
    plt.close()
    
    # Beeswarm Plot (Class 0)
    class_index = 0 
    class_name = target_le.classes_[class_index]
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[class_index], X_test, show=False)
    plt.tight_layout()
    plt.savefig(f'reports/figures/shap_beeswarm_{class_name}.png', dpi=300)
    plt.close()
    
    return explainer, shap_values, X_test, target_le, model

def generate_local_explanations(explainer, shap_values, X_test, target_le, model, data_dir='data/processed'):
    """Generates localized force plots for correct and erroneous instances[cite: 2]."""
    y_test_raw = pd.read_csv(f'{data_dir}/y_test.csv').values.ravel()
    y_test_num = target_le.transform(y_test_raw)
    y_pred = model.predict(X_test)
    
    correct_indices = np.where(y_pred == y_test_num)[0]
    error_indices = np.where(y_pred != y_test_num)[0]
    
    samples = [
        (correct_indices[0], "correct"),
        (correct_indices[1], "correct"),
        (error_indices[0], "error")
    ]
    
    for idx, status in samples:
        pred_class_idx = y_pred[idx]
        shap.force_plot(
            explainer.expected_value[pred_class_idx], 
            shap_values[pred_class_idx][idx], 
            X_test.iloc[idx], 
            matplotlib=True, 
            show=False
        )
        plt.savefig(f'reports/figures/shap_force_plot_idx{idx}_{status}.png', bbox_inches='tight', dpi=300)
        plt.close()
