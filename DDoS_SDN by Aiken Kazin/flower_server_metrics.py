#!/usr/bin/env python3
"""
Flower Server with Metrics Tracking

Enhanced Flower server that tracks metrics and sends them to the dashboard.

Usage:
    python flower_server_metrics.py --dashboard-url http://localhost:5000
"""

import argparse
import time
import requests
import numpy as np
import os
import socket
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MLflow for experiment tracking
try:
    import mlflow
    import mlflow.keras
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available - experiment tracking will be disabled")

# Import TensorFlow/Keras for model reconstruction and saving
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten, LSTM
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available - model saving will be disabled")

# Weights type - in newer Flower versions, weights are just List[np.ndarray]
Weights = List[np.ndarray]


class MetricsFedAvg(FedAvg):
    """FedAvg strategy with metrics tracking"""
    
    def __init__(self, dashboard_url: Optional[str] = None, num_rounds: int = 5, model_type: str = "Unknown", 
                 mlflow_tracking_uri: Optional[str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dashboard_url = dashboard_url
        self.num_rounds = num_rounds
        self.model_type = model_type
        self.current_round = 0
        self.round_start_time = None
        self.round_metrics = {
            "bytes_sent": 0,
            "bytes_received": 0,
            "params_sent": 0,
            "params_received": 0,
            "workers": set()
        }
        self.training_complete = False
        self.final_weights = None
        self.final_metrics = {}
        self.models_dir = "models"
        # Use /app/visualizations in container (mounted from ./visualizations), fallback to ./visualizations locally
        if os.path.exists("/app/visualizations"):
            self.visualizations_dir = "/app/visualizations"
        else:
            self.visualizations_dir = "visualizations"
        os.makedirs(self.visualizations_dir, exist_ok=True)
        logger.info(f"✅ Visualizations directory initialized: {os.path.abspath(self.visualizations_dir)}")
        # Model architecture parameters (defaults, will be updated from workers)
        self.num_features = 23  # Default from dataset
        self.num_classes = 2    # Default binary classification
        
        # Store confusion matrix data
        self.final_y_true = []
        self.final_y_pred = []
        self.iid_status = True  # Will be updated from workers
        
        # Initialize MLflow if available (non-blocking, will retry later)
        self.mlflow_run = None
        self.mlflow_enabled = False
        self.mlflow_tracking_uri = mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
        self.mlflow_experiment_name = "DDoS_Detection_FL"
        
        if MLFLOW_AVAILABLE:
            # Set tracking URI but don't verify connection yet (non-blocking)
            try:
                mlflow.set_tracking_uri(self.mlflow_tracking_uri)
                # Try to set up experiment with timeout (non-blocking)
                # If it fails, we'll retry later when actually logging
                try:
                    # Quick connection test with timeout
                    mlflow_host = self.mlflow_tracking_uri.replace("http://", "").replace("https://", "").split(":")[0]
                    mlflow_port = int(self.mlflow_tracking_uri.split(":")[-1].split("/")[0])
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(2)  # 2 second timeout
                    result = sock.connect_ex((mlflow_host, mlflow_port))
                    sock.close()
                    
                    if result == 0:
                        # MLflow server is reachable, try to set up experiment
                        experiment = mlflow.get_experiment_by_name(self.mlflow_experiment_name)
                        if experiment is None:
                            mlflow.create_experiment(self.mlflow_experiment_name)
                            logger.info(f"Created MLflow experiment: {self.mlflow_experiment_name}")
                        else:
                            mlflow.set_experiment(self.mlflow_experiment_name)
                            logger.info(f"Using MLflow experiment: {self.mlflow_experiment_name}")
                        self.mlflow_enabled = True
                    else:
                        logger.info(f"MLflow server not ready yet, will retry during training. Tracking URI: {self.mlflow_tracking_uri}")
                except Exception as e:
                    logger.info(f"MLflow server not ready yet ({e}), will retry during training. Tracking URI: {self.mlflow_tracking_uri}")
            except Exception as e:
                logger.warning(f"Could not initialize MLflow: {e}. MLflow tracking disabled.")
    
    def _send_to_dashboard(self, data: Dict):
        """Send metrics to dashboard"""
        if not self.dashboard_url:
            return
        
        try:
            response = requests.post(
                f"{self.dashboard_url}/api/update",
                json=data,
                timeout=1
            )
            if response.status_code != 200:
                logger.warning(f"Failed to send metrics to dashboard: {response.status_code}")
        except Exception as e:
            logger.debug(f"Dashboard not available: {e}")
    
    def _estimate_bytes(self, weights) -> int:
        """Estimate bytes for weights"""
        total_bytes = 0
        if weights is None:
            return 0
        # Handle different weight formats
        if isinstance(weights, list):
            for weight_item in weights:
                if isinstance(weight_item, bytes):
                    # Flower Parameters.tensors is a list of bytes
                    total_bytes += len(weight_item)
                elif isinstance(weight_item, np.ndarray):
                    total_bytes += weight_item.nbytes
                elif isinstance(weight_item, list):
                    for w in weight_item:
                        if isinstance(w, bytes):
                            total_bytes += len(w)
                        elif isinstance(w, np.ndarray):
                            total_bytes += w.nbytes
        return total_bytes
    
    def _count_parameters(self, weights) -> int:
        """Count total number of parameters (weights)"""
        total_params = 0
        if weights is None:
            return 0
        # Handle different weight formats
        if isinstance(weights, list):
            for weight_item in weights:
                if isinstance(weight_item, np.ndarray):
                    total_params += weight_item.size
                elif isinstance(weight_item, list):
                    for w in weight_item:
                        if isinstance(w, np.ndarray):
                            total_params += w.size
        return total_params
    
    def _create_model_architecture(self):
        """Reconstruct the model architecture based on model_type"""
        if not TF_AVAILABLE:
            logger.error("TensorFlow not available - cannot create model architecture")
            return None
        
        if self.model_type == "MLPv2":
            model = Sequential([
                tf.keras.Input(shape=(self.num_features,)),
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(32, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(self.num_classes, activation='softmax')
            ])
        elif self.model_type == "CNN1D":
            model = Sequential([
                Conv1D(64, kernel_size=3, activation='relu', input_shape=(self.num_features, 1)),
                BatchNormalization(),
                MaxPooling1D(2),
                Dropout(0.3),
                Conv1D(128, kernel_size=3, activation='relu'),
                BatchNormalization(),
                MaxPooling1D(2),
                Dropout(0.3),
                Flatten(),
                Dense(64, activation='relu'),
                Dense(self.num_classes, activation='softmax')
            ])
        elif self.model_type == "LSTM":
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(self.num_features, 1)),
                Dropout(0.3),
                LSTM(32),
                Dropout(0.3),
                Dense(self.num_classes, activation='softmax')
            ])
        elif self.model_type == "CNN_LSTM":
            model = Sequential([
                Conv1D(64, 3, activation='relu', input_shape=(self.num_features, 1)),
                MaxPooling1D(2),
                Dropout(0.3),
                LSTM(64, return_sequences=False),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dense(self.num_classes, activation='softmax')
            ])
        else:
            logger.warning(f"Unknown model type '{self.model_type}', defaulting to MLPv2")
            model = Sequential([
                tf.keras.Input(shape=(self.num_features,)),
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(32, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(self.num_classes, activation='softmax')
            ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _save_model(self):
        """Save the final aggregated model as H5 file (Keras format)"""
        if self.final_weights is None:
            logger.warning(f"No final weights to save for {self.model_type}")
            return
        
        if not TF_AVAILABLE:
            logger.warning(f"TensorFlow not available - cannot save model {self.model_type} as H5")
            return
        
        try:
            # Ensure models directory exists
            os.makedirs(self.models_dir, exist_ok=True)
            
            # Reconstruct model architecture
            model = self._create_model_architecture()
            if model is None:
                logger.error(f"Failed to create model architecture for {self.model_type}")
                return
            
            # Set the aggregated weights
            # self.final_weights is a list of numpy arrays (one per weight tensor)
            # We need to match weights to layers by their expected sizes
            # This handles cases where byte conversion might include extra padding
            
            # First, collect all expected weight sizes from the model
            expected_sizes = []
            for layer in model.layers:
                layer_weights = layer.get_weights()
                if layer_weights:
                    for w in layer_weights:
                        expected_sizes.append((np.prod(w.shape), w.shape))
            
            logger.debug(f"Total weights to assign: {len(self.final_weights)}, Expected layers: {len(expected_sizes)}")
            
            # Match weights by size (handle potential padding/extra bytes)
            used_indices = set()
            weight_idx = 0
            
            for layer in model.layers:
                layer_weights = layer.get_weights()
                if layer_weights:
                    layer_weight_list = []
                    
                    for i, expected_weight in enumerate(layer_weights):
                        expected_shape = expected_weight.shape
                        expected_size = np.prod(expected_shape)
                        
                        # Find matching weight by size (allowing for small differences due to padding)
                        matched = False
                        for idx, weight_array in enumerate(self.final_weights):
                            if idx in used_indices:
                                continue
                            
                            # Check if size matches (exact or within small tolerance for padding)
                            if weight_array.size == expected_size:
                                # Perfect match - reshape if needed
                                if weight_array.shape == expected_shape:
                                    layer_weight_list.append(weight_array)
                                else:
                                    layer_weight_list.append(weight_array.reshape(expected_shape))
                                used_indices.add(idx)
                                matched = True
                                break
                            elif weight_array.size > expected_size:
                                # Might have padding - take only the first expected_size elements
                                if weight_array.size >= expected_size:
                                    trimmed = weight_array[:expected_size].reshape(expected_shape)
                                    layer_weight_list.append(trimmed)
                                    used_indices.add(idx)
                                    matched = True
                                    logger.debug(f"Trimmed weight for {layer.name}[{i}]: {weight_array.size} -> {expected_size}")
                                    break
                        
                        if not matched:
                            # Fallback: use sequential assignment if size-based matching fails
                            if weight_idx < len(self.final_weights):
                                weight_array = self.final_weights[weight_idx]
                                if weight_array.size >= expected_size:
                                    if weight_array.size == expected_size:
                                        layer_weight_list.append(weight_array.reshape(expected_shape))
                                    else:
                                        # Trim excess
                                        layer_weight_list.append(weight_array[:expected_size].reshape(expected_shape))
                                    weight_idx += 1
                                else:
                                    logger.error(f"Cannot match weight for layer {layer.name}[{i}]: "
                                               f"expected {expected_size}, got {weight_array.size}")
                                    raise ValueError(f"Weight size mismatch for layer {layer.name}")
                            else:
                                logger.error(f"Not enough weights for layer {layer.name}[{i}]")
                                raise ValueError(f"Insufficient weights for layer {layer.name}")
                    
                    # Set weights for this layer
                    layer.set_weights(layer_weight_list)
            
            # Save as H5 file (Keras standard format)
            model_filename = f"{self.model_type}_FL.h5"
            model_path = os.path.join(self.models_dir, model_filename)
            
            model.save(model_path)
            
            num_params = model.count_params()
            accuracy = self.final_metrics.get('accuracy', 0)
            logger.info(f"💾 Model saved: {model_path} ({num_params:,} parameters, Accuracy: {accuracy:.4f})")
            
        except Exception as e:
            logger.error(f"Failed to save model {self.model_type}: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_confusion_matrix(self):
        """Generate and save confusion matrix as PNG image"""
        if not self.final_y_true or not self.final_y_pred:
            logger.warning(f"⚠️ No confusion matrix data available for {self.model_type} - y_true: {len(self.final_y_true) if self.final_y_true else 0}, y_pred: {len(self.final_y_pred) if self.final_y_pred else 0}")
            return
        
        logger.info(f"📊 Generating confusion matrix for {self.model_type} with {len(self.final_y_true)} samples...")
        
        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
            
            # Convert to numpy arrays
            y_true = np.array(self.final_y_true)
            y_pred = np.array(self.final_y_pred)
            
            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred) * 100
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot heatmap
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=[f"Class {i}" for i in range(len(cm))],
                yticklabels=[f"Class {i}" for i in range(len(cm))],
                cbar_kws={'label': 'Count'},
                ax=ax,
                linewidths=0.5,
                linecolor='gray'
            )
            
            # Set title and labels
            distribution_type = "IID" if self.iid_status else "Non-IID"
            ax.set_title(
                f"Confusion Matrix - {self.model_type} ({distribution_type})\n"
                f"Accuracy: {accuracy:.2f}% | Precision: {precision:.2f}% | Recall: {recall:.2f}% | F1: {f1:.2f}%",
                fontsize=13,
                fontweight='bold',
                pad=20
            )
            ax.set_xlabel("Predicted Label", fontsize=12)
            ax.set_ylabel("True Label", fontsize=12)
            
            plt.tight_layout()
            
            # Save figure
            filename = f"confusion_matrix_{self.model_type}_{distribution_type.lower()}.png"
            save_path = os.path.join(self.visualizations_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Confusion matrix saved: {save_path} (Accuracy: {accuracy:.2f}%)")
            logger.info(f"   File exists: {os.path.exists(save_path)}")
            logger.info(f"   File size: {os.path.getsize(save_path) if os.path.exists(save_path) else 'N/A'} bytes")
            
        except ImportError as e:
            logger.error(f"❌ Required libraries not available for confusion matrix visualization: {e}")
            import traceback
            traceback.print_exc()
        except Exception as e:
            logger.error(f"❌ Failed to save confusion matrix for {self.model_type}: {e}")
            import traceback
            traceback.print_exc()
    
    def configure_fit(self, server_round: int, parameters, client_manager):
        """Configure fit round - track round start and send params"""
        self.current_round = server_round
        self.round_start_time = time.time()
        self.round_metrics = {
            "bytes_sent": 0,
            "bytes_received": 0,
            "params_sent": 0,
            "params_received": 0,
            "workers": set()
        }
        
        # Start MLflow run on first round (retry initialization if needed)
        if server_round == 1 and MLFLOW_AVAILABLE:
            # Try to enable MLflow if not already enabled
            if not self.mlflow_enabled:
                try:
                    mlflow.set_tracking_uri(self.mlflow_tracking_uri)
                    experiment = mlflow.get_experiment_by_name(self.mlflow_experiment_name)
                    if experiment is None:
                        mlflow.create_experiment(self.mlflow_experiment_name)
                    else:
                        mlflow.set_experiment(self.mlflow_experiment_name)
                    self.mlflow_enabled = True
                    logger.info(f"MLflow enabled on round 1: {self.mlflow_experiment_name}")
                except Exception as e:
                    logger.debug(f"MLflow still not available: {e}. Continuing without MLflow tracking.")
            
            if self.mlflow_enabled:
                try:
                    run_name = f"{self.model_type}_FL_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    self.mlflow_run = mlflow.start_run(run_name=run_name)
                    
                    # Log FL configuration parameters
                    mlflow.log_params({
                        "model_type": self.model_type,
                        "num_rounds": self.num_rounds,
                        "min_fit_clients": self.min_fit_clients if hasattr(self, 'min_fit_clients') else 0,
                        "fraction_fit": self.fraction_fit if hasattr(self, 'fraction_fit') else 1.0,
                        "aggregation": "FedAvg"
                    })
                    logger.info(f"Started MLflow run: {run_name}")
                except Exception as e:
                    logger.warning(f"Failed to start MLflow run: {e}")
                    self.mlflow_enabled = False
        
        # Extract weights from parameters if needed
        # Flower Parameters.tensors is a list of bytes, not numpy arrays
        weights_bytes = []
        weights = []
        if hasattr(parameters, 'tensors'):
            weights_bytes = parameters.tensors
            # Convert bytes to numpy arrays for counting (assuming float32 = 4 bytes per param)
            for tensor_bytes in weights_bytes:
                if tensor_bytes and len(tensor_bytes) > 0:
                    weights.append(np.frombuffer(tensor_bytes, dtype=np.float32))
        elif parameters:
            weights_bytes = parameters if isinstance(parameters, list) else []
            weights = parameters if isinstance(parameters, list) else []
        
        # Estimate bytes sent (model weights to clients)
        bytes_sent = self._estimate_bytes(weights_bytes)
        params_sent = self._count_parameters(weights)
        self.round_metrics["bytes_sent"] = bytes_sent
        self.round_metrics["params_sent"] = params_sent
        
        # Send to dashboard
        self._send_to_dashboard({
            "round": server_round,
            "bytes_sent": bytes_sent,
            "params_sent": params_sent,
            "model_type": self.model_type
        })
        
        logger.info(f"Round {server_round}: Sending weights to clients (estimated {bytes_sent / 1024 / 1024:.2f} MB, {params_sent:,} params)")
        
        return super().configure_fit(server_round, parameters, client_manager)
    
    def aggregate_fit(self, rnd: int, results: List[Tuple], failures: List):
        """Aggregate fit results - track received params and aggregation"""
        # Track workers and their model types
        worker_models = {}
        model_params_count = 0
        for client, fit_res in results:
            worker_id = str(client.cid)
            self.round_metrics["workers"].add(worker_id)
            
            # Extract model type and params count from metrics if available
            model_type = "Unknown"
            if fit_res.metrics:
                model_type = fit_res.metrics.get("model_type", "Unknown")
                worker_models[worker_id] = model_type
                # Get model params count and architecture info from first worker
                if model_params_count == 0:
                    model_params_count = fit_res.metrics.get("model_params_count", 0)
                    # Extract num_features and num_classes from worker metrics for model reconstruction
                    worker_num_features = fit_res.metrics.get("num_features")
                    worker_num_classes = fit_res.metrics.get("num_classes")
                    if worker_num_features:
                        self.num_features = worker_num_features
                        logger.debug(f"Updated num_features from worker: {self.num_features}")
                    if worker_num_classes:
                        self.num_classes = worker_num_classes
                        logger.debug(f"Updated num_classes from worker: {self.num_classes}")
            
            # Forward DP metrics and IID status from worker to dashboard
            dp_metrics = {}
            iid_status = None
            if fit_res.metrics:
                if "dp_enabled" in fit_res.metrics:
                    dp_metrics["dp_enabled"] = fit_res.metrics["dp_enabled"]
                if "epsilon_this_round" in fit_res.metrics:
                    dp_metrics["epsilon_this_round"] = fit_res.metrics["epsilon_this_round"]
                if "epsilon_total" in fit_res.metrics:
                    dp_metrics["epsilon_total"] = fit_res.metrics["epsilon_total"]
                if "dp_clip_norm" in fit_res.metrics:
                    dp_metrics["dp_clip_norm"] = fit_res.metrics["dp_clip_norm"]
                if "dp_noise_multiplier" in fit_res.metrics:
                    dp_metrics["dp_noise_multiplier"] = fit_res.metrics["dp_noise_multiplier"]
                if "iid" in fit_res.metrics:
                    iid_status = fit_res.metrics["iid"]
                    self.iid_status = iid_status  # Store for visualizations
            
            dashboard_data = {
                "round": rnd,
                "worker_id": worker_id,
                "active": True,
                "model_type": model_type,
                **dp_metrics  # Include DP metrics if present
            }
            if iid_status is not None:
                dashboard_data["iid"] = iid_status
                logger.debug(f"Round {rnd}: Worker {worker_id} IID status: {iid_status}")
            
            self._send_to_dashboard(dashboard_data)
        
        # Send model params count to dashboard
        if model_params_count > 0:
            self._send_to_dashboard({
                "round": rnd,
                "model_type": self.model_type,
                "model_params_count": model_params_count
            })
        
        # Log model types being trained
        if worker_models:
            model_info = ", ".join([f"{wid}: {mtype}" for wid, mtype in worker_models.items()])
            logger.info(f"Round {rnd}: Training models - {model_info}")
        
        # Aggregate using parent class
        aggregated_weights, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        
        # Calculate bytes received
        total_bytes_received = 0
        total_params_received = 0
        for client, fit_res in results:
            if fit_res.parameters:
                weights_bytes = fit_res.parameters.tensors
                # Convert bytes to numpy arrays for counting (assuming float32 = 4 bytes per param)
                weights = []
                for tensor_bytes in weights_bytes:
                    if tensor_bytes and len(tensor_bytes) > 0:
                        weights.append(np.frombuffer(tensor_bytes, dtype=np.float32))
                total_bytes_received += self._estimate_bytes(weights_bytes)
                total_params_received += self._count_parameters(weights)
        
        self.round_metrics["bytes_received"] = total_bytes_received
        self.round_metrics["params_received"] = total_params_received
        
        # Calculate round time
        round_time = time.time() - self.round_start_time if self.round_start_time else 0
        
        # Send aggregation metrics to dashboard
        accuracy = aggregated_metrics.get("accuracy", 0.0) if aggregated_metrics else 0.0
        loss = aggregated_metrics.get("loss", 0.0) if aggregated_metrics else 0.0
        
        # Extract IID status from worker results (assume all workers have same IID setting)
        # Check all results to find IID status - don't default, let dashboard use its own default
        iid_status = None
        if results:
            for client, fit_res in results:
                if fit_res.metrics and "iid" in fit_res.metrics:
                    iid_status = bool(fit_res.metrics["iid"])  # Explicitly convert to bool
                    self.iid_status = iid_status  # Store for visualizations
                    logger.info(f"Round {rnd}: Extracted IID status from worker: {iid_status}")
                    break  # Use first found IID status (all workers should have same setting)
        
        dashboard_data = {
            "round": rnd,
            "bytes_received": total_bytes_received,
            "params_received": total_params_received,
            "round_time": round_time,
            "accuracy": float(accuracy),
            "loss": float(loss),
            "model_type": self.model_type
        }
        
        # Always include IID if we found it (don't force default)
        if iid_status is not None:
            dashboard_data["iid"] = iid_status
            logger.info(f"Round {rnd}: Sending IID status to dashboard: {iid_status}")
        else:
            logger.warning(f"Round {rnd}: No IID status found in worker metrics")
        
        # Forward DP metrics from worker results if available
        if results and results[0][1].metrics:
            worker_metrics = results[0][1].metrics
            if worker_metrics.get("dp_enabled"):
                dashboard_data.update({
                    "dp_enabled": True,
                    "epsilon_this_round": worker_metrics.get("epsilon_this_round", 0.0),
                    "epsilon_total": worker_metrics.get("epsilon_total", 0.0),
                    "dp_clip_norm": worker_metrics.get("dp_clip_norm", 0.0),
                    "dp_noise_multiplier": worker_metrics.get("dp_noise_multiplier", 0.0)
                })
            else:
                dashboard_data["dp_enabled"] = False
        
        self._send_to_dashboard(dashboard_data)
        
        # Get model types from results
        model_types = []
        for client, fit_res in results:
            if fit_res.metrics and "model_type" in fit_res.metrics:
                model_types.append(fit_res.metrics["model_type"])
        
        model_info = f" ({', '.join(set(model_types))})" if model_types else ""
        
        logger.info(
            f"Round {rnd}/{self.num_rounds} complete{model_info}: "
            f"Accuracy={accuracy:.4f}, Loss={loss:.4f}, "
            f"Time={round_time:.2f}s, "
            f"Received {total_bytes_received / 1024 / 1024:.2f} MB from {len(results)} clients"
        )
        
        # Log network metrics to MLflow (accuracy/loss will be logged in aggregate_evaluate)
        if self.mlflow_enabled and MLFLOW_AVAILABLE and self.mlflow_run:
            try:
                mlflow.log_metrics({
                    "round_time": round_time,
                    "bytes_sent": self.round_metrics["bytes_sent"],
                    "bytes_received": total_bytes_received,
                    "params_sent": self.round_metrics["params_sent"],
                    "params_received": total_params_received,
                    "num_workers": len(results)
                }, step=rnd)
                
                # Log DP metrics if available
                if results and results[0][1].metrics:
                    worker_metrics = results[0][1].metrics
                    if worker_metrics.get("dp_enabled"):
                        mlflow.log_metrics({
                            "epsilon_total": worker_metrics.get("epsilon_total", 0.0),
                            "dp_clip_norm": worker_metrics.get("dp_clip_norm", 0.0),
                            "dp_noise_multiplier": worker_metrics.get("dp_noise_multiplier", 0.0)
                        }, step=rnd)
            except Exception as e:
                logger.warning(f"Failed to log metrics to MLflow: {e}")
        
        # Store evaluation metrics for final logging (will be updated in aggregate_evaluate)
        self.round_metrics["accuracy"] = accuracy
        self.round_metrics["loss"] = loss
        
        # Check if training is complete
        if rnd >= self.num_rounds:
            self.training_complete = True
            # Convert aggregated_weights (Parameters object) to numpy arrays
            # Parameters.tensors is a list of bytes, convert to numpy arrays
            if hasattr(aggregated_weights, 'tensors'):
                # Convert bytes to numpy arrays (assuming float32, 4 bytes per value)
                weights_list = []
                for tensor_bytes in aggregated_weights.tensors:
                    if tensor_bytes and len(tensor_bytes) > 0:
                        # Convert bytes to numpy array (float32)
                        weight_array = np.frombuffer(tensor_bytes, dtype=np.float32)
                        weights_list.append(weight_array)
                self.final_weights = weights_list
            elif isinstance(aggregated_weights, list):
                # Already a list of numpy arrays
                self.final_weights = aggregated_weights
            else:
                logger.warning(f"Unexpected aggregated_weights type: {type(aggregated_weights)}")
                self.final_weights = []
            
            # Use evaluation metrics (from aggregate_evaluate) instead of fit metrics
            final_accuracy = self.round_metrics.get("accuracy", float(accuracy))
            final_loss = self.round_metrics.get("loss", float(loss))
            
            self.final_metrics = {
                "accuracy": final_accuracy,
                "loss": final_loss,
                "round": rnd,
                "total_rounds": self.num_rounds,
                "model_type": self.model_type,
                "timestamp": datetime.now().isoformat()
            }
            logger.info(f"✅ All {self.num_rounds} rounds completed for {self.model_type}! Training finished.")
            # Save the model as H5 file
            self._save_model()
            
            # Register model in MLflow and end run
            if self.mlflow_enabled and MLFLOW_AVAILABLE and self.mlflow_run:
                try:
                    model_path = os.path.join(self.models_dir, f"{self.model_type}_FL.h5")
                    if os.path.exists(model_path):
                        from tensorflow.keras.models import load_model
                        model = load_model(model_path)
                        mlflow.keras.log_model(
                            model,
                            "model",
                            registered_model_name=f"{self.model_type}_FL"
                        )
                        logger.info(f"Registered model in MLflow: {self.model_type}_FL")
                    
                    # Log final metrics (use evaluation metrics from round_metrics)
                    final_accuracy = self.final_metrics.get("accuracy", 0.0)
                    final_loss = self.final_metrics.get("loss", 0.0)
                    mlflow.log_metrics({
                        "final_accuracy": float(final_accuracy),
                        "final_loss": float(final_loss),
                        "total_rounds": self.num_rounds
                    })
                    logger.info(f"Logged final metrics to MLflow: accuracy={final_accuracy:.4f}, loss={final_loss:.4f}")
                    
                    mlflow.end_run()
                    logger.info("MLflow run completed")
                except Exception as e:
                    logger.error(f"Failed to register model in MLflow: {e}")
                    if mlflow.active_run():
                        mlflow.end_run()
        
        return aggregated_weights, aggregated_metrics
    
    def configure_evaluate(self, server_round: int, parameters, client_manager):
        """Configure evaluate round"""
        # Extract weights from parameters if needed
        weights_bytes = []
        if hasattr(parameters, 'tensors'):
            weights_bytes = parameters.tensors
        elif parameters:
            weights_bytes = parameters if isinstance(parameters, list) else []
        
        bytes_sent = self._estimate_bytes(weights_bytes)
        # Count parameters for evaluate round too
        weights = []
        for tensor_bytes in weights_bytes:
            if tensor_bytes and len(tensor_bytes) > 0:
                weights.append(np.frombuffer(tensor_bytes, dtype=np.float32))
        params_sent = self._count_parameters(weights)
        self._send_to_dashboard({
            "round": server_round,
            "bytes_sent": bytes_sent,
            "params_sent": params_sent,
            "model_type": self.model_type
        })
        return super().configure_evaluate(server_round, parameters, client_manager)
    
    def aggregate_evaluate(self, rnd: int, results: List[Tuple], failures: List):
        """Aggregate evaluate results - track evaluation metrics"""
        # Get model types and accuracies from evaluation results
        model_types = []
        accuracies = []
        total_samples = 0
        
        for client, eval_res in results:
            if eval_res.metrics:
                if "model_type" in eval_res.metrics:
                    model_types.append(eval_res.metrics["model_type"])
                # Extract accuracy from worker metrics (workers return "test_accuracy")
                if "test_accuracy" in eval_res.metrics:
                    accuracies.append((eval_res.metrics["test_accuracy"], eval_res.num_examples))
                    total_samples += eval_res.num_examples
        
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(rnd, results, failures)
        
        # Calculate weighted average accuracy from worker results
        accuracy = 0.0
        if accuracies and total_samples > 0:
            accuracy = sum(acc * samples for acc, samples in accuracies) / total_samples
        elif aggregated_metrics:
            # Fallback: try to get accuracy from aggregated metrics
            accuracy = aggregated_metrics.get("accuracy", aggregated_metrics.get("test_accuracy", 0.0))
        
        model_info = f" ({', '.join(set(model_types))})" if model_types else ""
        logger.info(f"Round {rnd} evaluation complete{model_info}: Accuracy={accuracy:.4f}, Loss={aggregated_loss:.4f}")
        
        # Update round metrics with evaluation results
        self.round_metrics["accuracy"] = accuracy
        self.round_metrics["loss"] = aggregated_loss
        
        # Collect predictions for confusion matrix (only on final round)
        y_true_all = []
        y_pred_all = []
        if rnd >= self.num_rounds:
            logger.info(f"🔍 Final round ({rnd}) - Collecting confusion matrix data from {len(results)} workers...")
            for client, eval_res in results:
                if eval_res.metrics:
                    if "y_true" in eval_res.metrics and "y_pred" in eval_res.metrics:
                        y_true_count = len(eval_res.metrics["y_true"])
                        y_pred_count = len(eval_res.metrics["y_pred"])
                        y_true_all.extend(eval_res.metrics["y_true"])
                        y_pred_all.extend(eval_res.metrics["y_pred"])
                        logger.info(f"  ✓ Worker {client.cid}: Collected {y_true_count} true labels, {y_pred_count} predictions")
                    else:
                        logger.warning(f"  ✗ Worker {client.cid}: No y_true/y_pred in metrics. Available keys: {list(eval_res.metrics.keys())}")
                else:
                    logger.warning(f"  ✗ Worker {client.cid}: No metrics available")
            
            # Store for saving confusion matrix image
            if y_true_all and y_pred_all:
                self.final_y_true = y_true_all
                self.final_y_pred = y_pred_all
                logger.info(f"✅ Collected {len(y_true_all)} total predictions for confusion matrix ({self.model_type})")
                # Generate and save confusion matrix image
                self._save_confusion_matrix()
            else:
                logger.error(f"❌ No confusion matrix data collected for {self.model_type} on final round! y_true: {len(y_true_all)}, y_pred: {len(y_pred_all)}")
        
        dashboard_data = {
            "round": rnd,
            "accuracy": float(accuracy),
            "loss": float(aggregated_loss),
            "model_type": self.model_type
        }
        
        # Add confusion matrix data if available
        if y_true_all and y_pred_all:
            dashboard_data["confusion_matrix"] = {
                "y_true": y_true_all,
                "y_pred": y_pred_all
            }
        
        self._send_to_dashboard(dashboard_data)
        
        # Log evaluation metrics to MLflow (accuracy/loss from evaluation)
        if self.mlflow_enabled and MLFLOW_AVAILABLE and self.mlflow_run:
            try:
                mlflow.log_metrics({
                    "round_accuracy": float(accuracy),
                    "round_loss": float(aggregated_loss),
                    "accuracy": float(accuracy),  # Also log as "accuracy" for consistency
                    "loss": float(aggregated_loss)  # Also log as "loss" for consistency
                }, step=rnd)
                logger.debug(f"Logged evaluation metrics to MLflow for round {rnd}: accuracy={accuracy:.4f}, loss={aggregated_loss:.4f}")
            except Exception as e:
                logger.warning(f"Failed to log evaluation metrics to MLflow: {e}")
        
        return aggregated_loss, aggregated_metrics


def main():
    parser = argparse.ArgumentParser(description='Flower Federated Learning Server with Metrics')
    parser.add_argument('--address', type=str, default='0.0.0.0', help='Server address')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--min-clients', type=int, default=1, help='Minimum clients')
    parser.add_argument('--num-rounds', type=int, default=1, help='Number of rounds')
    parser.add_argument('--fraction-fit', type=float, default=1.0, help='Fraction fit')
    parser.add_argument('--fraction-evaluate', type=float, default=1.0, help='Fraction evaluate')
    parser.add_argument('--dashboard-url', type=str, default='http://localhost:5000', 
                       help='Dashboard URL for metrics (default: http://localhost:5000)')
    parser.add_argument('--model-type', type=str, default='Unknown',
                       help='Model type being trained (for logging)')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory to save trained models (default: models)')
    
    args = parser.parse_args()
    
    # Create strategy with metrics tracking
    strategy = MetricsFedAvg(
        dashboard_url=args.dashboard_url,
        num_rounds=args.num_rounds,
        model_type=args.model_type,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
    )
    # Set models directory
    strategy.models_dir = args.models_dir
    
    # Set round timeout to 10 minutes (600 seconds) to prevent indefinite waiting
    # This allows workers enough time to train while preventing hangs
    config = ServerConfig(num_rounds=args.num_rounds, round_timeout=600.0)
    
    print("\n" + "="*60)
    print(" Flower Federated Learning Server (with Metrics)")
    print("="*60)
    print(f" Address:        {args.address}")
    print(f" Port:           {args.port}")
    print(f" Min Clients:    {args.min_clients}")
    print(f" Rounds:         {args.num_rounds}")
    print(f" Model Type:     {args.model_type}")
    print(f" Dashboard:      {args.dashboard_url}")
    print("="*60)
    print("\n Waiting for clients to connect...")
    print(" Server will auto-shutdown after training completes")
    print(" Press Ctrl+C to stop the server\n")
    
    try:
        fl.server.start_server(
            server_address=f"{args.address}:{args.port}",
            config=config,
            strategy=strategy,
        )
        logger.info(f"✅ Server for {args.model_type} completed all rounds. Shutting down...")
        import sys
        sys.exit(0)
    except KeyboardInterrupt:
        logger.info("\n Server stopped by user")
        import sys
        sys.exit(0)
    except Exception as e:
        logger.error(f" Server error: {e}")
        import sys
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

