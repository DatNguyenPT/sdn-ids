"""
Worker Model Manager

Manages models on workers for federated inference:
- Receive and store models
- Version management
- Model loading and caching
"""

import os
import shutil
import logging
from typing import Dict, Optional, List
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import threading

logger = logging.getLogger(__name__)


class WorkerModelManager:
    """Manage models on worker for local inference"""
    
    def __init__(self, models_dir: str = "/app/models", worker_id: str = "worker1"):
        """
        Initialize worker model manager.
        
        Args:
            models_dir: Directory to store models on worker
            worker_id: Worker identifier
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.worker_id = worker_id
        self.models_cache: Dict[str, Model] = {}
        self.model_versions: Dict[str, str] = {}
        self.lock = threading.Lock()
    
    def load_model(
        self,
        model_type: str,
        model_path: Optional[str] = None,
        version: str = "latest",
        cache: bool = True
    ) -> Optional[Model]:
        """
        Load model for inference.
        
        Args:
            model_type: Model type (MLPv2, LSTM, CNN1D, CNN_LSTM)
            model_path: Path to model file (if None, auto-detect)
            version: Model version
            cache: Whether to cache loaded model
            
        Returns:
            Loaded Keras model or None if failed
        """
        # Auto-detect model path if not provided
        if model_path is None:
            # Try shared volume path first (Docker)
            shared_path = f"/app/models/{model_type}_FL.h5"
            if os.path.exists(shared_path):
                model_path = shared_path
            else:
                # Try local models directory
                local_path = self.models_dir / f"{model_type}_FL.h5"
                if local_path.exists():
                    model_path = str(local_path)
                else:
                    logger.error(f"❌ Model file not found for {model_type}")
                    return None
        
        # Check cache
        cache_key = f"{model_type}_{version}"
        if cache and cache_key in self.models_cache:
            logger.info(f"✅ Using cached model: {model_type} (v{version})")
            return self.models_cache[cache_key]
        
        # Load model
        try:
            with self.lock:
                logger.info(f"📥 Loading model: {model_type} from {model_path}")
                model = load_model(model_path, compile=False)
                
                # Compile model for inference
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Cache model
                if cache:
                    self.models_cache[cache_key] = model
                    self.model_versions[model_type] = version
                
                logger.info(f"✅ Model {model_type} loaded successfully (v{version})")
                return model
                
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_type}: {e}")
            return None
    
    def get_model_info(self, model_type: str) -> Dict:
        """Get information about loaded model"""
        cache_key = f"{model_type}_{self.model_versions.get(model_type, 'latest')}"
        
        if cache_key in self.models_cache:
            model = self.models_cache[cache_key]
            return {
                "model_type": model_type,
                "version": self.model_versions.get(model_type, "latest"),
                "loaded": True,
                "parameters": model.count_params(),
                "input_shape": str(model.input_shape),
                "output_shape": str(model.output_shape)
            }
        else:
            return {
                "model_type": model_type,
                "version": self.model_versions.get(model_type, "unknown"),
                "loaded": False
            }
    
    def list_available_models(self) -> List[str]:
        """List available model files"""
        models = []
        
        # Check shared volume (Docker)
        shared_dir = Path("/app/models")
        if shared_dir.exists():
            for model_file in shared_dir.glob("*_FL.h5"):
                model_type = model_file.stem.replace("_FL", "")
                models.append(model_type)
        
        # Check local directory
        if self.models_dir.exists():
            for model_file in self.models_dir.glob("*_FL.h5"):
                model_type = model_file.stem.replace("_FL", "")
                if model_type not in models:
                    models.append(model_type)
        
        return models
    
    def clear_cache(self, model_type: Optional[str] = None):
        """Clear model cache"""
        with self.lock:
            if model_type:
                # Clear specific model
                keys_to_remove = [k for k in self.models_cache.keys() if k.startswith(model_type)]
                for key in keys_to_remove:
                    del self.models_cache[key]
                if model_type in self.model_versions:
                    del self.model_versions[model_type]
                logger.info(f"🗑️ Cleared cache for {model_type}")
            else:
                # Clear all
                self.models_cache.clear()
                self.model_versions.clear()
                logger.info("🗑️ Cleared all model cache")

