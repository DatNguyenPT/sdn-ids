"""
Model Distribution Service for Federated Learning

Distributes final trained models from server to workers for federated inference.
"""

import os
import shutil
import logging
from typing import List, Dict, Optional
from pathlib import Path
import requests

logger = logging.getLogger(__name__)


class ModelDistributor:
    """Distribute trained models to workers after FL training completes"""
    
    def __init__(self, models_dir: str = "models", worker_base_url: str = None):
        """
        Initialize model distributor.
        
        Args:
            models_dir: Directory containing trained models
            worker_base_url: Base URL for worker inference APIs (optional)
        """
        self.models_dir = Path(models_dir)
        self.worker_base_url = worker_base_url
        self.distribution_log = []
    
    def distribute_model(
        self,
        model_type: str,
        model_path: Optional[str] = None,
        workers: Optional[List[str]] = None,
        version: str = "latest"
    ) -> Dict:
        """
        Distribute model to workers.
        
        Args:
            model_type: Model type (MLPv2, LSTM, CNN1D, CNN_LSTM)
            model_path: Path to model file (if None, auto-detect)
            workers: List of worker IDs to distribute to (if None, distribute to all)
            version: Model version string
            
        Returns:
            Distribution status dictionary
        """
        # Auto-detect model path if not provided
        if model_path is None:
            model_path = self.models_dir / f"{model_type}_FL.h5"
        
        if not os.path.exists(model_path):
            logger.error(f"❌ Model file not found: {model_path}")
            return {
                "success": False,
                "error": f"Model file not found: {model_path}",
                "distributed_to": []
            }
        
        logger.info(f"📦 Distributing {model_type} model (version {version}) to workers...")
        
        # In Docker environment, models are shared via volume
        # Workers can access models from shared volume
        shared_model_path = f"/app/models/{model_type}_FL.h5"
        
        distribution_status = {
            "success": True,
            "model_type": model_type,
            "version": version,
            "model_path": str(model_path),
            "shared_path": shared_model_path,
            "distributed_to": [],
            "distribution_method": "shared_volume"  # Docker volume sharing
        }
        
        # If workers list provided, notify them via API
        if workers and self.worker_base_url:
            for worker_id in workers:
                try:
                    # Notify worker about new model
                    response = requests.post(
                        f"{self.worker_base_url}/{worker_id}/model/update",
                        json={
                            "model_type": model_type,
                            "model_path": shared_model_path,
                            "version": version
                        },
                        timeout=5
                    )
                    if response.status_code == 200:
                        distribution_status["distributed_to"].append(worker_id)
                        logger.info(f"✅ Notified worker {worker_id} about model update")
                    else:
                        logger.warning(f"⚠️ Failed to notify worker {worker_id}: {response.status_code}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not notify worker {worker_id}: {e}")
        
        # Log distribution
        self.distribution_log.append({
            "model_type": model_type,
            "version": version,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "distributed_to": distribution_status["distributed_to"]
        })
        
        logger.info(f"✅ Model {model_type} (v{version}) distributed successfully")
        return distribution_status
    
    def get_distribution_history(self) -> List[Dict]:
        """Get history of model distributions"""
        return self.distribution_log
    
    def check_model_availability(self, model_type: str) -> bool:
        """Check if model file exists"""
        model_path = self.models_dir / f"{model_type}_FL.h5"
        return model_path.exists()


class SharedVolumeDistributor(ModelDistributor):
    """
    Distributor for Docker shared volumes.
    
    In Docker Compose, models are shared via volume mount.
    Workers can directly access models from shared volume.
    """
    
    def distribute_model(
        self,
        model_type: str,
        model_path: Optional[str] = None,
        workers: Optional[List[str]] = None,
        version: str = "latest"
    ) -> Dict:
        """
        Distribute model via shared volume (Docker).
        
        Models are already accessible to workers via volume mount.
        This method just verifies and logs distribution.
        """
        if model_path is None:
            model_path = self.models_dir / f"{model_type}_FL.h5"
        
        if not os.path.exists(model_path):
            logger.error(f"❌ Model file not found: {model_path}")
            return {
                "success": False,
                "error": f"Model file not found: {model_path}",
                "distributed_to": []
            }
        
        # In Docker, models are shared via volume
        # Workers can access from: /app/models/{MODEL_TYPE}_FL.h5
        shared_path = f"/app/models/{model_type}_FL.h5"
        
        logger.info(f"✅ Model {model_type} available to workers via shared volume: {shared_path}")
        
        return {
            "success": True,
            "model_type": model_type,
            "version": version,
            "model_path": str(model_path),
            "shared_path": shared_path,
            "distributed_to": workers or ["all_workers"],
            "distribution_method": "shared_volume",
            "message": "Model accessible via Docker volume mount"
        }

