import pandas as pd
import numpy as np
import os
import argparse

try:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve, classification_report
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Error importing machine learning libraries: {e}")
    print("Please install the required dependencies using:")
    print("  pip install lightgbm scikit-learn matplotlib pandas numpy")
    exit(1)

def train_and_evaluate(dataset_csv, output_dir="."):
    if not os.path.exists(dataset_csv):
        raise FileNotFoundError(f"ML dataset not found: {dataset_csv}")
        
    print(f"Loading ML dataset: {dataset_csv}")
    df = pd.read_csv(dataset_csv)
    
    # Check size
    if len(df) < 5:
        raise ValueError(f"Dataset is too small ({len(df)} rows) to perform training.")
        
    features = [
        'liquidity_type', 'liquidity_strength', 'sweep_direction', 'sweep_size',
        'bos_strength', 'fvg_rejected', 'retracement_depth', 'time_since_sweep',
        'ny_session', 'london_session', 'asian_session'
    ]
    
    # Verify features are in dataframe
    for f in features + ['label']:
        if f not in df.columns:
            raise ValueError(f"Required column '{f}' is missing from the dataset.")
            
    X = df[features]
    y = df['label']
    
    # Chronological Split (first 80% train, last 20% test) - No Shuffle!
    split_idx = int(len(df) * 0.8)
    if split_idx == 0:
        split_idx = 1
        
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Data Split Details:")
    print(f"  - Train Set Size: {len(X_train)} rows")
    print(f"  - Test Set Size: {len(X_test)} rows")
    print(f"  - Train Class Distribution (Success rate): {y_train.mean():.2%}")
    print(f"  - Test Class Distribution (Success rate): {y_test.mean():.2%}")
    
    # Define and train LightGBM classifier
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        random_state=42,
        min_child_samples=3,
        verbosity=-1
    )
    
    print("Training LightGBM model...")
    model.fit(X_train, y_train)
    
    # Predict probabilities and classes
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # 1. Evaluate ROC-AUC
    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"\n======================================")
    print(f"Evaluation Metrics (Test Set):")
    print(f"======================================")
    print(f"ROC-AUC Score: {auc:.4f}")
    
    # 2. Precision & Recall
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    
    # 3. Feature Importance
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
    print("\nFeature Importances (splits count):")
    for feat, imp in feat_imp.items():
        print(f"  - {feat}: {imp}")
        
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save plots
    # Plot 1: Feature Importance
    plt.figure(figsize=(10, 6))
    feat_imp.plot(kind='barh', color='skyblue').invert_yaxis()
    plt.title("LightGBM Feature Importance (Sequence Quality)")
    plt.xlabel("Importance (Split Count)")
    plt.tight_layout()
    feat_imp_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(feat_imp_path)
    plt.close()
    print(f"\nSaved feature importance plot to: {feat_imp_path}")
    
    # Plot 2: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved ROC curve plot to: {roc_path}")
    
    # Plot 3: Calibration Curve
    # Use 5 bins for calibration curve
    prob_true, prob_pred = calibration_curve(y_test, y_pred_prob, n_bins=5, strategy='uniform')
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='LightGBM')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives (Actual Success Rate)')
    plt.title('Calibration Curve (Reliability Diagram)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    cal_path = os.path.join(output_dir, "calibration_curve.png")
    plt.savefig(cal_path)
    plt.close()
    print(f"Saved calibration curve plot to: {cal_path}")
    print("======================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train LightGBM Classifier on MNQ Trade Sequences")
    parser.add_argument('--dataset', type=str, required=True, help="Path to processed mnq_ml_dataset.csv")
    parser.add_argument('--outdir', type=str, default=".", help="Directory to save evaluation plots")
    args = parser.parse_args()
    
    try:
        train_and_evaluate(args.dataset, args.outdir)
    except Exception as e:
        print(f"Training error: {e}")
        exit(1)
