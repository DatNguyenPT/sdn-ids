"""
Tests for data validation module
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlops.data_validation import DataValidator


class TestDataValidator:
    """Test data validation functionality"""
    
    def test_valid_dataset(self):
        """Test validation with valid dataset"""
        # Create valid test dataset
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'label': [0, 1, 0, 1, 0]
        })
        
        validator = DataValidator()
        is_valid, results = validator.validate_dataset(df)
        
        assert is_valid == True
        assert results['structure']['passed'] == True
        assert results['labels']['passed'] == True
    
    def test_missing_label(self):
        """Test validation with missing label column"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [0.1, 0.2, 0.3]
        })
        
        validator = DataValidator()
        is_valid, results = validator.validate_dataset(df)
        
        assert is_valid == False
        assert results['structure']['passed'] == False
    
    def test_invalid_labels(self):
        """Test validation with invalid label values"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'label': [0, 1, 2]  # Invalid: 2 is not binary
        })
        
        validator = DataValidator()
        is_valid, results = validator.validate_dataset(df)
        
        assert is_valid == False
        assert results['labels']['passed'] == False
    
    def test_null_labels(self):
        """Test validation with null labels"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'label': [0, 1, None]  # Null label
        })
        
        validator = DataValidator()
        is_valid, results = validator.validate_dataset(df)
        
        assert is_valid == False
        assert 'null' in str(results['labels']['issues']).lower()
    
    def test_empty_dataset(self):
        """Test validation with empty dataset"""
        df = pd.DataFrame()
        
        validator = DataValidator()
        is_valid, results = validator.validate_dataset(df)
        
        assert is_valid == False
    
    def test_class_balance_detection(self):
        """Test class balance detection"""
        # Imbalanced dataset
        df = pd.DataFrame({
            'feature1': [1] * 100 + [2] * 10,
            'label': [0] * 100 + [1] * 10  # 10:1 ratio
        })
        
        validator = DataValidator()
        is_valid, results = validator.validate_dataset(df)
        
        balance = results['class_balance']
        assert balance['balance_ratio'] < 0.3  # Severely imbalanced
        assert balance['is_balanced'] == False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

