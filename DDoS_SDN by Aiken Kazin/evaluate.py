import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Prevent GUI pop-up
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from tensorflow.keras.models import load_model

# Import preprocessing pipeline
from preprocess import process_col, normalization, split_dataset


def evaluate_model(model_name=None):
    # === Resolve model path ===
    models_dir = "models"
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"❌ Models directory not found: {models_dir}")

    if model_name:
        model_path = os.path.join(models_dir, model_name)
    else:
        # Auto-detect latest model by modification time
        h5_files = [f for f in os.listdir(models_dir) if f.endswith(".h5")]
        if not h5_files:
            raise FileNotFoundError("❌ No .h5 models found in /models directory.")
        latest_file = max(
            [os.path.join(models_dir, f) for f in h5_files],
            key=os.path.getmtime
        )
        model_path = latest_file
        model_name = os.path.basename(model_path)

    if not os.path.exists(model_path):
        available = [f for f in os.listdir(models_dir) if f.endswith(".h5")]
        raise FileNotFoundError(
            f"❌ Model '{model_name}' not found in {models_dir}.\n"
            f"Available models: {available}"
        )

    print(f" Loading trained model: {model_name}")
    model = load_model(model_path)
    print(f" Model loaded successfully from {model_path}")

    # === Load and preprocess dataset ===
    print("\n Loading dataset for evaluation...")
    df = pd.read_csv("dataset_sdn.csv")
    df.columns = df.columns.str.strip().str.lower()

    df = process_col(df)
    df = normalization(df)
    X_train, X_test, y_train, y_test = split_dataset(df)

    # === Ensure consistent feature order ===
    feature_path = "models/feature_order.csv"
    if os.path.exists(feature_path):
        expected_cols = pd.read_csv(feature_path).squeeze().tolist()
        X_test = X_test.reindex(columns=expected_cols, fill_value=0)
        print(" Aligned test features with training order")
    else:
        pd.Series(X_test.columns).to_csv(feature_path, index=False)
        print(" Saved feature order for future consistency")

    print(f" Model expects input shape: {model.input_shape}")
    print(f" X_test shape: {X_test.shape}")

    # === Evaluate ===
    print("\n Running evaluation...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print("\n Evaluation complete!")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall   : {rec:.4f}")
    print(f"   F1-score : {f1:.4f}\n")

    # === Create output directories ===
    os.makedirs("img", exist_ok=True)
    os.makedirs("report", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # === Confusion Matrix ===
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    cm_path = os.path.join("img", f"confusion_matrix_{model_name}_{timestamp}.png")
    plt.savefig(cm_path)
    plt.close()
    print(f" Confusion matrix saved to {cm_path}")

    # === Classification Report ===
    cls_report = classification_report(y_test, y_pred, output_dict=True)
    cls_df = pd.DataFrame(cls_report).transpose()

    # === Save Excel Report ===
    report_path = os.path.join("report", f"evaluation_report_{model_name}_{timestamp}.xlsx")
    with pd.ExcelWriter(report_path) as writer:
        summary_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
            "Value": [acc, prec, rec, f1]
        })
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
        cls_df.to_excel(writer, sheet_name="Class_Report")

    print(f" Evaluation report saved to {report_path}")

    # === Return for automation ===
    return {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "report_path": report_path,
        "confusion_matrix_path": cm_path
    }


def main():
    try:
        # Support optional CLI argument for model name
        model_name = sys.argv[1] if len(sys.argv) > 1 else None
        results = evaluate_model(model_name)

        print("\n Final Results:")
        for k, v in results.items():
            if k not in ["report_path", "confusion_matrix_path", "model"]:
                print(f"   {k:<10}: {v:.4f}")
        print(f"\n Report : {results['report_path']}")
        print(f"  Matrix : {results['confusion_matrix_path']}")
        print("\n Evaluation pipeline finished successfully!\n")

    except Exception as e:
        print(f"\n Error during evaluation: {e}")


if __name__ == "__main__":
    main()
