"""
Data Quality Monitoring for FL Pipeline
Monitors data drift and quality changes over time
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime
import logging
import json
import os

logger = logging.getLogger(__name__)


class DataQualityMonitor:
    """Monitors data quality and detects drift"""
    
    def __init__(self, baseline_path: Optional[str] = None):
        """
        Initialize data quality monitor
        
        Args:
            baseline_path: Path to save/load baseline statistics
        """
        self.baseline_stats = None
        self.baseline_path = baseline_path or "mlops/baseline_stats.json"
        self.drift_threshold = 0.1  # 10% drift threshold
    
    def compute_baseline(self, df: pd.DataFrame, save: bool = True) -> Dict:
        """
        Compute baseline statistics for data quality checks
        
        Args:
            df: DataFrame to use as baseline
            save: Whether to save baseline to file
            
        Returns:
            Dictionary of baseline statistics
        """
        logger.info("Computing baseline statistics...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.baseline_stats = {
            "timestamp": datetime.now().isoformat(),
            "row_count": len(df),
            "column_count": len(df.columns),
            "numeric_features": {},
            "categorical_features": {},
            "null_counts": df.isnull().sum().to_dict(),
            "class_distribution": {}
        }
        
        # Compute statistics for numeric features
        for col in numeric_cols:
            if col != 'label':
                self.baseline_stats["numeric_features"][col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "median": float(df[col].median()),
                    "q25": float(df[col].quantile(0.25)),
                    "q75": float(df[col].quantile(0.75))
                }
        
        # Compute statistics for categorical features
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        for col in categorical_cols:
            if col != 'label':
                value_counts = df[col].value_counts().to_dict()
                self.baseline_stats["categorical_features"][col] = {
                    "unique_count": int(df[col].nunique()),
                    "top_values": dict(list(value_counts.items())[:5])  # Top 5 values
                }
        
        # Class distribution
        if 'label' in df.columns:
            class_dist = df['label'].value_counts().to_dict()
            total = len(df)
            self.baseline_stats["class_distribution"] = {
                int(k): {
                    "count": int(v),
                    "percentage": float(v / total * 100)
                }
                for k, v in class_dist.items()
            }
        
        # Save baseline if requested
        if save:
            self._save_baseline()
        
        logger.info(f"Baseline computed: {len(df)} rows, {len(numeric_cols)} numeric features")
        return self.baseline_stats
    
    def load_baseline(self) -> bool:
        """Load baseline statistics from file"""
        if not os.path.exists(self.baseline_path):
            logger.warning(f"Baseline file not found: {self.baseline_path}")
            return False
        
        try:
            with open(self.baseline_path, 'r') as f:
                self.baseline_stats = json.load(f)
            logger.info(f"Baseline loaded from {self.baseline_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load baseline: {e}")
            return False
    
    def _save_baseline(self):
        """Save baseline statistics to file"""
        os.makedirs(os.path.dirname(self.baseline_path), exist_ok=True)
        try:
            with open(self.baseline_path, 'w') as f:
                json.dump(self.baseline_stats, f, indent=2)
            logger.info(f"Baseline saved to {self.baseline_path}")
        except Exception as e:
            logger.error(f"Failed to save baseline: {e}")
    
    def detect_drift(self, new_df: pd.DataFrame, threshold: Optional[float] = None) -> Dict:
        """
        Detect data drift compared to baseline
        
        Args:
            new_df: New DataFrame to compare
            threshold: Drift threshold (default: self.drift_threshold)
            
        Returns:
            Dictionary of drift detection results
        """
        if self.baseline_stats is None:
            raise ValueError("Baseline not computed. Call compute_baseline() first.")
        
        threshold = threshold or self.drift_threshold
        
        drift_results = {
            "timestamp": datetime.now().isoformat(),
            "threshold": threshold,
            "drift_detected": False,
            "feature_drift": {},
            "class_drift": {},
            "warnings": []
        }
        
        # Check feature distributions
        baseline_numeric = self.baseline_stats.get("numeric_features", {})
        new_numeric_cols = new_df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in new_numeric_cols:
            if col != 'label' and col in baseline_numeric:
                baseline_mean = baseline_numeric[col]["mean"]
                new_mean = new_df[col].mean()
                
                # Calculate drift ratio
                if abs(baseline_mean) > 1e-6:
                    drift_ratio = abs(new_mean - baseline_mean) / abs(baseline_mean)
                else:
                    drift_ratio = abs(new_mean - baseline_mean)
                
                if drift_ratio > threshold:
                    drift_results["drift_detected"] = True
                    drift_results["feature_drift"][col] = {
                        "baseline_mean": baseline_mean,
                        "new_mean": float(new_mean),
                        "drift_ratio": float(drift_ratio),
                        "drift_percentage": float(drift_ratio * 100)
                    }
        
        # Check class distribution drift
        baseline_class_dist = self.baseline_stats.get("class_distribution", {})
        if 'label' in new_df.columns and baseline_class_dist:
            new_class_dist = new_df['label'].value_counts().to_dict()
            new_total = len(new_df)
            baseline_total = sum([v["count"] for v in baseline_class_dist.values()])
            
            for label in baseline_class_dist.keys():
                baseline_ratio = baseline_class_dist[label]["percentage"] / 100.0
                new_count = new_class_dist.get(int(label), 0)
                new_ratio = new_count / new_total if new_total > 0 else 0.0
                
                drift_ratio = abs(new_ratio - baseline_ratio)
                if drift_ratio > threshold:
                    drift_results["drift_detected"] = True
                    drift_results["class_drift"][f"class_{label}"] = {
                        "baseline_ratio": baseline_ratio,
                        "new_ratio": float(new_ratio),
                        "drift_ratio": float(drift_ratio),
                        "baseline_count": baseline_class_dist[label]["count"],
                        "new_count": int(new_count)
                    }
        
        # Check row count change
        baseline_row_count = self.baseline_stats.get("row_count", 0)
        new_row_count = len(new_df)
        if baseline_row_count > 0:
            row_count_change = abs(new_row_count - baseline_row_count) / baseline_row_count
            if row_count_change > threshold:
                drift_results["warnings"].append(
                    f"Row count changed significantly: "
                    f"{baseline_row_count} → {new_row_count} "
                    f"({row_count_change*100:.1f}% change)"
                )
        
        return drift_results
    
    def compare_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict:
        """
        Compare two datasets and report differences
        
        Args:
            df1: First dataset (baseline)
            df2: Second dataset (new)
            
        Returns:
            Comparison results
        """
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "differences": {},
            "warnings": []
        }
        
        # Compare shapes
        if df1.shape != df2.shape:
            comparison["differences"]["shape"] = {
                "df1": df1.shape,
                "df2": df2.shape
            }
            comparison["warnings"].append("Dataset shapes differ")
        
        # Compare columns
        df1_cols = set(df1.columns)
        df2_cols = set(df2.columns)
        if df1_cols != df2_cols:
            missing_in_df2 = df1_cols - df2_cols
            missing_in_df1 = df2_cols - df1_cols
            comparison["differences"]["columns"] = {
                "missing_in_df2": list(missing_in_df2),
                "missing_in_df1": list(missing_in_df1)
            }
            comparison["warnings"].append("Column sets differ")
        
        # Compare class distributions
        if 'label' in df1.columns and 'label' in df2.columns:
            dist1 = df1['label'].value_counts().to_dict()
            dist2 = df2['label'].value_counts().to_dict()
            comparison["differences"]["class_distribution"] = {
                "df1": dist1,
                "df2": dist2
            }
        
        return comparison

