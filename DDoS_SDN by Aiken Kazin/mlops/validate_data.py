#!/usr/bin/env python3
"""
Validate Dataset Before FL Training

This script validates the dataset and reports any issues.

Usage:
    python mlops/validate_data.py [dataset_path]
"""

import sys
import os
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlops.data_validation import DataValidator


def main():
    """Validate dataset"""
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "dataset_sdn.csv"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset not found at {dataset_path}")
        return 1
    
    print(f"📊 Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    df.columns = df.columns.str.strip().str.lower()
    
    print(f"   Dataset shape: {df.shape}\n")
    
    # Validate
    print("🔍 Validating dataset...")
    validator = DataValidator()
    is_valid, results = validator.validate_dataset(df)
    
    # Print summary
    print("\n" + validator.get_validation_summary())
    
    if is_valid:
        print("\n✅ Dataset validation PASSED - ready for FL training")
        return 0
    else:
        print("\n❌ Dataset validation FAILED - please fix issues before training")
        return 1


if __name__ == "__main__":
    exit(main())

