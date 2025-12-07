"""
Model Registry for Federated Learning Models
Handles model versioning, registration, and retrieval using MLflow
"""

import os
import mlflow
import mlflow.keras
from datetime import datetime
from typing import Optional, Dict, Any
from tensorflow.keras.models import load_model
import logging

logger = logging.getLogger(__name__)


class FLModelRegistry:
    """Model registry for managing FL model versions"""
    
    def __init__(self, tracking_uri: str = "http://mlflow-server:5000"):
        """
        Initialize MLflow model registry
        
        Args:
            tracking_uri: MLflow tracking server URI
        """
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient()
        self.experiment_name = "DDoS_Detection_FL"
        
        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created new MLflow experiment: {self.experiment_name}")
            else:
                mlflow.set_experiment(self.experiment_name)
                logger.info(f"Using existing MLflow experiment: {self.experiment_name}")
        except Exception as e:
            logger.warning(f"Could not set up MLflow experiment: {e}. Using default.")
    
    def register_model_version(
        self,
        model_path: str,
        model_type: str,
        round_num: int,
        accuracy: float,
        loss: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a new model version after FL round
        
        Args:
            model_path: Path to saved model file (.h5)
            model_type: Type of model (MLPv2, LSTM, CNN1D, CNN_LSTM)
            round_num: FL round number
            accuracy: Model accuracy
            loss: Model loss
            metadata: Additional metadata (bytes sent/received, etc.)
        
        Returns:
            Version name/ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_name = f"{model_type}_FL_Round{round_num}_Acc{accuracy:.3f}_{timestamp}"
        model_name = f"{model_type}_FL"
        
        try:
            # Start MLflow run
            run = mlflow.start_run(run_name=version_name)
            
            # Log parameters
            params = {
                "model_type": model_type,
                "round": round_num,
                "version": version_name,
                "timestamp": timestamp
            }
            if metadata:
                params.update(metadata)
            mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics({
                "accuracy": accuracy,
                "loss": loss,
                "round": round_num
            })
            
            # Log model artifact
            if os.path.exists(model_path):
                mlflow.keras.log_model(
                    load_model(model_path),
                    "model",
                    registered_model_name=model_name
                )
                logger.info(f"Registered model version: {version_name}")
            else:
                logger.warning(f"Model file not found: {model_path}")
            
            mlflow.end_run()
            return version_name
            
        except Exception as e:
            logger.error(f"Error registering model version: {e}")
            if mlflow.active_run():
                mlflow.end_run()
            raise
    
    def get_latest_model(self, model_type: str, stage: str = "None"):
        """
        Get latest registered model version
        
        Args:
            model_type: Type of model (MLPv2, LSTM, CNN1D, CNN_LSTM)
            stage: Model stage (None, Staging, Production)
        
        Returns:
            Latest model version or None
        """
        try:
            model_name = f"{model_type}_FL"
            latest_versions = self.client.get_latest_versions(
                model_name, stages=[stage]
            )
            if latest_versions:
                return latest_versions[0]
            return None
        except Exception as e:
            logger.error(f"Error getting latest model: {e}")
            return None
    
    def promote_to_production(self, model_type: str, version: str):
        """
        Promote model version to production
        
        Args:
            model_type: Type of model
            version: Model version to promote
        """
        try:
            model_name = f"{model_type}_FL"
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
                archive_existing_versions=True
            )
            logger.info(f"Promoted {model_name} version {version} to Production")
        except Exception as e:
            logger.error(f"Error promoting model to production: {e}")
            raise
    
    def log_fl_round(
        self,
        model_type: str,
        round_num: int,
        accuracy: float,
        loss: float,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Log FL round metrics to MLflow
        
        Args:
            model_type: Type of model
            round_num: FL round number
            accuracy: Round accuracy
            loss: Round loss
            metrics: Additional metrics (bytes_sent, bytes_received, etc.)
        """
        try:
            run_name = f"{model_type}_FL_Round_{round_num}"
            
            # Log metrics
            round_metrics = {
                "round_accuracy": accuracy,
                "round_loss": loss,
                "round": round_num
            }
            if metrics:
                round_metrics.update(metrics)
            
            mlflow.log_metrics(round_metrics, step=round_num)
            logger.debug(f"Logged FL round {round_num} metrics for {model_type}")
            
        except Exception as e:
            logger.error(f"Error logging FL round: {e}")

