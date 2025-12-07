"""
MLOps utilities for Federated Learning DDoS Detection System
"""

__version__ = "1.0.0"

# Import key modules for easy access
try:
    from mlops.model_registry import FLModelRegistry
    from mlops.data_validation import DataValidator
    from mlops.data_quality import DataQualityMonitor
    
    __all__ = [
        'FLModelRegistry',
        'DataValidator',
        'DataQualityMonitor'
    ]
except ImportError:
    # Modules may not be available in all environments
    __all__ = []

