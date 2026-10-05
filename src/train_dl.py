import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, brier_score_loss, roc_curve, confusion_matrix)
from sklearn.calibration import calibration_curve
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

from model import build_mlp
from search import CONFIGS
from features import ComplexFeatureEncoder, QuantilePiecewiseLinearEncoder, encode

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def plot_history(history, save_path_prefix):
    # Plot loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{save_path_prefix}_loss.png')
    plt.close()
    
    # Plot accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{save_path_prefix}_accuracy.png')
    plt.close()

def plot_roc(y_true, y_prob, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Receiver Operating Characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def plot_calibration(y_true, y_prob, save_path):
    # Calibration matters for a risk predictor because we want predicted probabilities (e.g., 0.8) to match the actual likelihood of cancer (80% of such patients have cancer).
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)
    
    plt.figure()
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="MLP")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.title('Calibration Curve (Reliability Diagram)')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.legend()
    # Adding a comment about why calibration matters for risk prediction
    plt.text(0.05, 0.9, "Calibration is crucial: a predicted risk of 0.8\nshould mean 80% observed cancer rate.", fontsize=9, bbox=dict(facecolor='white', alpha=0.5))
    plt.savefig(save_path)
    plt.close()

def train_and_evaluate_final():
    # Load splits
    train_data = np.load('data_splits/train.npz')
    val_data = np.load('data_splits/val.npz')
    test_data = np.load('data_splits/test.npz')
    
    X_train, y_train = train_data['X'], train_data['y']
    X_val, y_val = val_data['X'], val_data['y']
    X_test, y_test = test_data['X'], test_data['y']
    
    # Compute class weights
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight = {cls: weight for cls, weight in zip(classes, weights)}
    
    # Read best config from arch_search_v2.csv
    if not os.path.exists('results/arch_search_v2.csv'):
        raise FileNotFoundError("results/arch_search_v2.csv not found. Run search.py first.")
        
    arch_df = pd.read_csv('results/arch_search_v2.csv')
    encoded_df = arch_df[arch_df['Encoded'] == True]
    best_config_name = encoded_df.loc[encoded_df['Mean Val AUC'].idxmax()]['Config']
        
    print(f"Using best config: {best_config_name}")
    config = CONFIGS[best_config_name]
    
    is_encoded = config.pop('encoded')
    use_cw = config.pop('use_class_weight')
    cw = class_weight if use_cw else None
    
    if is_encoded:
        import joblib
        encoder = joblib.load('models/feature_encoder.pkl')
        train_raw = np.load('data_splits/train_raw.npz')
        val_raw = np.load('data_splits/val_raw.npz')
        test_raw = np.load('data_splits/test_raw.npz')
        
        X_train = encode(train_raw['X'], encoder)
        X_val = encode(val_raw['X'], encoder)
        X_test = encode(test_raw['X'], encoder)
    
    input_dim = X_train.shape[1]
    
    seeds = [1, 2, 3, 4, 5]
    results = []
    
    best_val_loss_overall = float('inf')
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    for seed in seeds:
        print(f"--- Training with seed {seed} ---")
        set_seeds(seed)
        tf.keras.backend.clear_session()
        
        model = build_mlp(input_dim=input_dim, **config)
        
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True, verbose=0
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            class_weight=cw,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Best val loss for this seed
        val_loss = min(history.history['val_loss'])
        
        if val_loss < best_val_loss_overall:
            best_val_loss_overall = val_loss
            model.save('models/cancer_dl_model_v2.keras')
            # Plot only for the best seed
            plot_history(history, 'results/dl_best')
            
            y_test_prob_best = model.predict(X_test, verbose=0).flatten()
            y_test_pred_best = (y_test_prob_best > 0.5).astype(int)
            plot_roc(y_test, y_test_prob_best, 'results/dl_roc.png')
            plot_confusion_matrix(y_test, y_test_pred_best, 'results/dl_cm.png')
            plot_calibration(y_test, y_test_prob_best, 'results/dl_calibration.png')
            
        
        # Evaluate on TEST set
        y_test_prob = model.predict(X_test, verbose=0).flatten()
        y_test_pred = (y_test_prob > 0.5).astype(int)
        
        acc = accuracy_score(y_test, y_test_pred)
        prec = precision_score(y_test, y_test_pred)
        rec = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        auc = roc_auc_score(y_test, y_test_prob)
        brier = brier_score_loss(y_test, y_test_prob)
        
        results.append({
            'Seed': seed,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'ROC-AUC': auc,
            'Brier Score': brier
        })
        
    results_df = pd.DataFrame(results)
    
    # Calculate mean and std
    summary = results_df.drop(columns=['Seed']).agg(['mean', 'std']).T
    summary.reset_index(inplace=True)
    summary.rename(columns={'index': 'Metric'}, inplace=True)
    
    # Format mean +/- std string
    summary['Mean +/- Std'] = summary.apply(lambda row: f"{row['mean']:.4f} +/- {row['std']:.4f}", axis=1)
    
    # Save per-seed results and mean/std summary
    with open('results/dl_results.csv', 'w') as f:
        f.write("--- Per Seed Results ---\n")
        results_df.to_csv(f, index=False)
        f.write("\n--- Mean and Std ---\n")
        summary.to_csv(f, index=False)
        
    print("Final training complete. Saved models and plots.")

if __name__ == "__main__":
    train_and_evaluate_final()
