#!/usr/bin/env python3
"""
Setup Baseline Statistics for Data Quality Monitoring

This script computes and saves baseline statistics from the dataset
for data drift detection in FL training.

Usage:
    python mlops/setup_baseline.py
"""

import sys
import os
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlops.data_quality import DataQualityMonitor
from preprocess import process_col, normalization


def main():
    """Compute baseline statistics from dataset"""
    dataset_path = "dataset_sdn.csv"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset not found at {dataset_path}")
        return 1
    
    print(f"📊 Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    df.columns = df.columns.str.strip().str.lower()
    
    print(f"   Dataset shape: {df.shape}")
    
    # Preprocess (same as FL pipeline)
    print("🔧 Preprocessing dataset...")
    df = process_col(df)
    df = normalization(df)
    
    print(f"   After preprocessing: {df.shape}")
    
    # Compute baseline
    print("📈 Computing baseline statistics...")
    monitor = DataQualityMonitor()
    baseline = monitor.compute_baseline(df, save=True)
    
    print("\n✅ Baseline statistics computed and saved!")
    print(f"   Baseline path: {monitor.baseline_path}")
    print(f"   Timestamp: {baseline['timestamp']}")
    print(f"   Rows: {baseline['row_count']}")
    print(f"   Columns: {baseline['column_count']}")
    print(f"   Numeric features: {len(baseline['numeric_features'])}")
    
    if 'class_distribution' in baseline:
        print("\n📊 Class Distribution:")
        for label, info in baseline['class_distribution'].items():
            print(f"   Class {label}: {info['count']} ({info['percentage']:.2f}%)")
    
    return 0


if __name__ == "__main__":
    exit(main())

