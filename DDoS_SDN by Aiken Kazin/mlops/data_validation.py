"""
Data Validation Module for FL Pipeline
Validates dataset before FL training to ensure data quality
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates dataset before FL training"""
    
    def __init__(self):
        self.expectations = []
        self.validation_results = {}
    
    def validate_dataset(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Validate dataset before FL training
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, validation_results)
        """
        if df is None or df.empty:
            return False, {"error": "Dataset is empty or None"}
        
        results = {}
        all_passed = True
        
        # 1. Basic structure checks
        structure_check = self._check_structure(df)
        results["structure"] = structure_check
        if not structure_check["passed"]:
            all_passed = False
        
        # 2. Label validation
        label_check = self._check_labels(df)
        results["labels"] = label_check
        if not label_check["passed"]:
            all_passed = False
        
        # 3. Feature validation
        feature_check = self._check_features(df)
        results["features"] = feature_check
        if not feature_check["passed"]:
            all_passed = False
        
        # 4. Data quality checks
        quality_check = self._check_data_quality(df)
        results["quality"] = quality_check
        if not quality_check["passed"]:
            all_passed = False
        
        # 5. Class balance check
        balance_check = self._check_class_balance(df)
        results["class_balance"] = balance_check
        
        self.validation_results = results
        return all_passed, results
    
    def _check_structure(self, df: pd.DataFrame) -> Dict:
        """Check basic dataset structure"""
        checks = {
            "passed": True,
            "row_count": len(df),
            "column_count": len(df.columns),
            "issues": []
        }
        
        if len(df) == 0:
            checks["passed"] = False
            checks["issues"].append("Dataset has no rows")
        
        if len(df.columns) == 0:
            checks["passed"] = False
            checks["issues"].append("Dataset has no columns")
        
        # Check for required columns (label column)
        if 'label' not in df.columns:
            checks["passed"] = False
            checks["issues"].append("Missing required 'label' column")
        
        return checks
    
    def _check_labels(self, df: pd.DataFrame) -> Dict:
        """Validate label column"""
        checks = {
            "passed": True,
            "issues": []
        }
        
        if 'label' not in df.columns:
            checks["passed"] = False
            checks["issues"].append("Label column not found")
            return checks
        
        # Check for null labels
        null_count = df['label'].isnull().sum()
        if null_count > 0:
            checks["passed"] = False
            checks["issues"].append(f"Found {null_count} null labels")
        
        # Check label values (should be binary: 0 or 1)
        unique_labels = df['label'].dropna().unique()
        invalid_labels = [l for l in unique_labels if l not in [0, 1]]
        if invalid_labels:
            checks["passed"] = False
            checks["issues"].append(f"Invalid label values: {invalid_labels}")
        
        checks["unique_labels"] = sorted(unique_labels.tolist())
        checks["label_counts"] = df['label'].value_counts().to_dict()
        
        return checks
    
    def _check_features(self, df: pd.DataFrame) -> Dict:
        """Validate feature columns"""
        checks = {
            "passed": True,
            "feature_count": 0,
            "numeric_features": 0,
            "categorical_features": 0,
            "issues": []
        }
        
        # Exclude label column
        feature_cols = [c for c in df.columns if c != 'label']
        checks["feature_count"] = len(feature_cols)
        
        if len(feature_cols) == 0:
            checks["passed"] = False
            checks["issues"].append("No feature columns found")
            return checks
        
        # Count numeric vs categorical
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
        
        checks["numeric_features"] = len(numeric_cols)
        checks["categorical_features"] = len(categorical_cols)
        
        # Check for features with all null values
        for col in feature_cols:
            if df[col].isnull().all():
                checks["issues"].append(f"Column '{col}' has all null values")
                # Don't fail validation, just warn
        
        # Check for constant features (no variance)
        for col in numeric_cols:
            if df[col].nunique() <= 1:
                checks["issues"].append(f"Column '{col}' has no variance (constant values)")
        
        return checks
    
    def _check_data_quality(self, df: pd.DataFrame) -> Dict:
        """Check data quality issues"""
        checks = {
            "passed": True,
            "null_counts": {},
            "duplicate_rows": 0,
            "issues": []
        }
        
        # Check null values
        null_counts = df.isnull().sum()
        checks["null_counts"] = null_counts[null_counts > 0].to_dict()
        
        if len(checks["null_counts"]) > 0:
            total_nulls = null_counts.sum()
            null_percentage = (total_nulls / (len(df) * len(df.columns))) * 100
            if null_percentage > 10:  # More than 10% nulls
                checks["issues"].append(f"High null percentage: {null_percentage:.2f}%")
                # Don't fail, just warn
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        checks["duplicate_rows"] = int(duplicate_count)
        if duplicate_count > 0:
            duplicate_percentage = (duplicate_count / len(df)) * 100
            if duplicate_percentage > 5:  # More than 5% duplicates
                checks["issues"].append(f"High duplicate percentage: {duplicate_percentage:.2f}%")
        
        # Check for infinite values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                checks["issues"].append(f"Column '{col}' contains infinite values")
                checks["passed"] = False
        
        return checks
    
    def _check_class_balance(self, df: pd.DataFrame) -> Dict:
        """Check class distribution balance"""
        checks = {
            "passed": True,
            "class_distribution": {},
            "balance_ratio": 0.0,
            "is_balanced": True,
            "warnings": []
        }
        
        if 'label' not in df.columns:
            checks["passed"] = False
            return checks
        
        class_counts = df['label'].value_counts().sort_index()
        checks["class_distribution"] = class_counts.to_dict()
        
        if len(class_counts) < 2:
            checks["warnings"].append("Only one class present in dataset")
            checks["is_balanced"] = False
            return checks
        
        # Calculate balance ratio (minority/majority)
        min_count = class_counts.min()
        max_count = class_counts.max()
        balance_ratio = min_count / max_count if max_count > 0 else 0.0
        
        checks["balance_ratio"] = float(balance_ratio)
        
        # Warn if severely imbalanced (< 0.3 ratio)
        if balance_ratio < 0.3:
            checks["is_balanced"] = False
            checks["warnings"].append(
                f"Severe class imbalance detected (ratio: {balance_ratio:.3f}). "
                f"Consider using class weights or resampling."
            )
        elif balance_ratio < 0.5:
            checks["warnings"].append(
                f"Moderate class imbalance (ratio: {balance_ratio:.3f})"
            )
        
        return checks
    
    def get_validation_summary(self) -> str:
        """Get human-readable validation summary"""
        if not self.validation_results:
            return "No validation performed yet"
        
        summary_lines = ["Data Validation Summary:"]
        summary_lines.append("=" * 50)
        
        # Structure
        structure = self.validation_results.get("structure", {})
        summary_lines.append(f"Rows: {structure.get('row_count', 0)}")
        summary_lines.append(f"Columns: {structure.get('column_count', 0)}")
        
        # Labels
        labels = self.validation_results.get("labels", {})
        if labels.get("passed"):
            label_counts = labels.get("label_counts", {})
            summary_lines.append(f"Labels: {label_counts}")
        else:
            summary_lines.append(f"Labels: FAILED - {labels.get('issues', [])}")
        
        # Features
        features = self.validation_results.get("features", {})
        summary_lines.append(f"Features: {features.get('feature_count', 0)} "
                           f"(numeric: {features.get('numeric_features', 0)}, "
                           f"categorical: {features.get('categorical_features', 0)})")
        
        # Class balance
        balance = self.validation_results.get("class_balance", {})
        balance_ratio = balance.get("balance_ratio", 0.0)
        summary_lines.append(f"Class Balance Ratio: {balance_ratio:.3f}")
        
        # Issues
        all_issues = []
        for check_name, check_result in self.validation_results.items():
            if isinstance(check_result, dict) and "issues" in check_result:
                all_issues.extend([f"{check_name}: {issue}" for issue in check_result["issues"]])
        
        if all_issues:
            summary_lines.append("\nIssues Found:")
            for issue in all_issues:
                summary_lines.append(f"  - {issue}")
        
        return "\n".join(summary_lines)

