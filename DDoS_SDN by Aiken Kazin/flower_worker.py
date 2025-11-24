#!/usr/bin/env python3
"""
Flower Worker/Agent for Federated Learning

This script creates a Flower client that participates in federated learning:
- Receives model weights from the server
- Trains locally on its data partition
- Sends updated weights back to the server
- Participates in multiple federated learning rounds

Usage:
    python flower_worker.py --server-address localhost:8080 --worker-id worker1
    python flower_worker.py --server-address 192.168.1.50:8080 --worker-id worker2 --data-partition 0.5
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, Any, Tuple, Optional
import flwr as fl
from tensorflow.keras.models import Sequential
# Weights type - in newer Flower versions, weights are List[np.ndarray]
from typing import List
Weights = List[np.ndarray]
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten, LSTM
from tensorflow.keras.optimizers import Adam
import logging

# Import preprocessing pipeline
try:
    from preprocess import process_col, normalization, split_dataset
except ImportError:
    print("Warning: preprocess module not found. Make sure you're running from the correct directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)


class FlowerWorker(fl.client.NumPyClient):
    """Flower worker that trains models locally and participates in federated learning"""
    
    def __init__(
        self,
        worker_id: str,
        data_partition: float = 1.0,
        model_type: str = "MLPv2",
        epochs_per_round: int = 5,
        batch_size: int = 32,
        total_rounds: int = 5
    ):
        """
        Initialize the Flower worker.
        
        Args:
            worker_id: Unique identifier for this worker
            data_partition: Fraction of dataset to use (0.0-1.0)
            model_type: Type of model to train (MLPv2, CNN1D, LSTM, CNN_LSTM)
            epochs_per_round: Number of epochs to train per federated round
            batch_size: Batch size for training
            total_rounds: Total number of FL rounds expected
        """
        self.worker_id = worker_id
        self.data_partition = data_partition
        self.model_type = model_type
        self.epochs_per_round = epochs_per_round
        self.batch_size = batch_size
        self.total_rounds = total_rounds
        self.current_round = 0
        self.training_complete = False
        
        # Load and preprocess data
        logger.info(f"[{worker_id}] Loading dataset...")
        self._load_data()
        
        # Initialize model
        self.model = None
        self.num_features = self.X_train.shape[1]
        self.num_classes = len(np.unique(self.y_train))
        
        logger.info(f"[{worker_id}] Worker initialized")
        logger.info(f"  - Data partition: {data_partition*100:.1f}% ({len(self.X_train)} samples)")
        logger.info(f"  - Features: {self.num_features}, Classes: {self.num_classes}")
        logger.info(f"  - Model type: {model_type}")
        logger.info(f"  - Total rounds: {total_rounds}")
    
    def _load_data(self):
        """Load and preprocess the dataset"""
        # Load dataset
        if not os.path.exists("dataset_sdn.csv"):
            raise FileNotFoundError("dataset_sdn.csv not found in current directory")
        
        df = pd.read_csv("dataset_sdn.csv")
        df.columns = df.columns.str.strip().str.lower()
        
        # Preprocess
        df = process_col(df)
        df = normalization(df)
        
        # Split dataset
        X_train, X_test, y_train, y_test = split_dataset(df)
        
        # Partition data (simulate different data distributions per worker)
        if self.data_partition < 1.0:
            # Use a subset of data
            n_samples = int(len(X_train) * self.data_partition)
            indices = np.random.RandomState(seed=hash(self.worker_id) % 2**32).choice(
                len(X_train), size=n_samples, replace=False
            )
            # Use iloc to select rows by position, not column names
            # X_train is a DataFrame, y_train is a Series from train_test_split
            X_train = X_train.iloc[indices].reset_index(drop=True)
            if isinstance(y_train, pd.Series):
                y_train = y_train.iloc[indices].reset_index(drop=True)
            else:
                # If it's a numpy array
                y_train = y_train[indices]
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # Prepare 3D data for CNN/LSTM models (add channel dimension)
        self.X_train_3d = np.expand_dims(self.X_train, axis=2) if self.model_type in ['CNN1D', 'LSTM', 'CNN_LSTM'] else None
        self.X_test_3d = np.expand_dims(self.X_test, axis=2) if self.model_type in ['CNN1D', 'LSTM', 'CNN_LSTM'] else None
        
        # Convert to categorical
        self.y_train_cat = tf.keras.utils.to_categorical(self.y_train)
        self.y_test_cat = tf.keras.utils.to_categorical(self.y_test)
    
    def _create_model(self) -> Sequential:
        """Create the model architecture"""
        logger.info(f"[{self.worker_id}] Creating {self.model_type} model architecture...")
        
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
            logger.info(f"[{self.worker_id}] MLPv2 model created: Input({self.num_features}) -> Dense(64) -> Dense(32) -> Output({self.num_classes})")
            
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
            logger.info(f"[{self.worker_id}] CNN1D model created: Input({self.num_features},1) -> Conv1D(64) -> Conv1D(128) -> Dense(64) -> Output({self.num_classes})")
            
        elif self.model_type == "LSTM":
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(self.num_features, 1)),
                Dropout(0.3),
                LSTM(32),
                Dropout(0.3),
                Dense(self.num_classes, activation='softmax')
            ])
            logger.info(f"[{self.worker_id}] LSTM model created: Input({self.num_features},1) -> LSTM(64) -> LSTM(32) -> Output({self.num_classes})")
            
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
            logger.info(f"[{self.worker_id}] CNN_LSTM model created: Input({self.num_features},1) -> Conv1D(64) -> LSTM(64) -> Dense(64) -> Output({self.num_classes})")
            
        else:
            logger.warning(f"[{self.worker_id}] Unknown model type '{self.model_type}', defaulting to MLPv2")
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
        
        # Log model summary
        total_params = model.count_params()
        logger.info(f"[{self.worker_id}] Model compiled: {total_params:,} total parameters")
        
        return model
    
    def get_parameters(self, config: Dict[str, Any]) -> Weights:
        """Return current model parameters"""
        if self.model is None:
            self.model = self._create_model()
        
        # Return weights as a list of numpy arrays (Flower format)
        weights = []
        for layer in self.model.layers:
            layer_weights = layer.get_weights()
            if layer_weights:
                weights.extend(layer_weights)
        return weights
    
    def set_parameters(self, weights: Weights):
        """Set model parameters from server"""
        if self.model is None:
            self.model = self._create_model()
        
        # Set weights layer by layer
        # Weights come as a flat list, need to group by layer
        weight_idx = 0
        for layer in self.model.layers:
            layer_weights = layer.get_weights()
            if layer_weights:
                # Get the number of weight arrays for this layer
                num_layer_weights = len(layer_weights)
                # Extract weights for this layer
                layer_weight_list = weights[weight_idx:weight_idx + num_layer_weights]
                layer.set_weights(layer_weight_list)
                weight_idx += num_layer_weights
    
    def fit(self, parameters: Weights, config: Dict[str, Any]) -> Tuple[Weights, int, Dict[str, Any]]:
        """
        Train the model locally and return updated weights.
        
        This is called by the server during federated training rounds.
        """
        # Track current round from config if available
        if "server_round" in config:
            self.current_round = config["server_round"]
        
        logger.info(f"[{self.worker_id}] Starting local training (Model: {self.model_type}, Round: {self.current_round}/{self.total_rounds})...")
        
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Train locally - use 3D data for CNN/LSTM models
        train_data = self.X_train_3d if self.model_type in ['CNN1D', 'LSTM', 'CNN_LSTM'] else self.X_train
        
        history = self.model.fit(
            train_data,
            self.y_train_cat,
            epochs=self.epochs_per_round,
            batch_size=self.batch_size,
            verbose=0  # Set to 1 for verbose output
        )
        
        # Get updated weights
        updated_weights = self.get_parameters(config)
        
        # Get training metrics
        train_loss = float(history.history['loss'][-1])
        train_accuracy = float(history.history['accuracy'][-1])
        
        num_samples = len(self.X_train)
        
        # Get model parameter count
        model_params_count = self.model.count_params() if self.model else 0
        
        logger.info(
            f"[{self.worker_id}] Training complete (Model: {self.model_type}, Round: {self.current_round}/{self.total_rounds}) - "
            f"Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, "
            f"Samples: {num_samples}, Epochs: {self.epochs_per_round}, Params: {model_params_count:,}"
        )
        
        # Check if this is the last round
        if self.current_round >= self.total_rounds:
            self.training_complete = True
            logger.info(f"[{self.worker_id}] ✅ All {self.total_rounds} rounds completed! Training finished.")
        
        return updated_weights, num_samples, {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "worker_id": self.worker_id,
            "model_type": self.model_type,
            "current_round": self.current_round,
            "model_params_count": model_params_count,
            "num_features": self.num_features,
            "num_classes": self.num_classes
        }
    
    def evaluate(self, parameters: Weights, config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluate the model on local test data.
        
        This is called by the server during federated evaluation rounds.
        """
        logger.info(f"[{self.worker_id}] Evaluating model (Model: {self.model_type})...")
        
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Evaluate - use 3D data for CNN/LSTM models
        test_data = self.X_test_3d if self.model_type in ['CNN1D', 'LSTM', 'CNN_LSTM'] else self.X_test
        
        loss, accuracy = self.model.evaluate(
            test_data,
            self.y_test_cat,
            batch_size=self.batch_size,
            verbose=0
        )
        
        num_samples = len(self.X_test)
        
        logger.info(
            f"[{self.worker_id}] Evaluation complete (Model: {self.model_type}) - "
            f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, "
            f"Samples: {num_samples}"
        )
        
        return float(loss), num_samples, {
            "test_loss": float(loss),
            "test_accuracy": float(accuracy),
            "worker_id": self.worker_id,
            "model_type": self.model_type
        }


def main():
    parser = argparse.ArgumentParser(description='Flower Worker/Agent for Federated Learning')
    parser.add_argument(
        '--server-address',
        type=str,
        default='localhost:8080',
        help='Flower server address (default: localhost:8080)'
    )
    parser.add_argument(
        '--worker-id',
        type=str,
        required=True,
        help='Unique worker identifier (e.g., worker1, worker2)'
    )
    parser.add_argument(
        '--data-partition',
        type=float,
        default=1.0,
        help='Fraction of dataset to use (0.0-1.0, default: 1.0)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='MLPv2',
        choices=['MLPv2', 'CNN1D', 'LSTM', 'CNN_LSTM'],
        help='Model architecture type (default: MLPv2)'
    )
    parser.add_argument(
        '--epochs-per-round',
        type=int,
        default=5,
        help='Number of epochs per federated round (default: 5)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    parser.add_argument(
        '--total-rounds',
        type=int,
        default=5,
        help='Total number of FL rounds (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Validate data partition
    if not 0.0 < args.data_partition <= 1.0:
        raise ValueError("data_partition must be between 0.0 and 1.0")
    
    print("\n" + "="*60)
    print(" Flower Worker/Agent")
    print("="*60)
    print(f" Worker ID:       {args.worker_id}")
    print(f" Server Address:  {args.server_address}")
    print(f" Data Partition:  {args.data_partition*100:.1f}%")
    print(f" Model Type:      {args.model_type} ⭐")
    print(f" Epochs/Round:    {args.epochs_per_round}")
    print(f" Batch Size:      {args.batch_size}")
    print(f" Total Rounds:    {args.total_rounds}")
    print("="*60)
    print(f"\n🚀 Training Model: {args.model_type}")
    print(" Connecting to Flower server...")
    print(" Worker will auto-shutdown after training completes\n")
    
    try:
        # Create worker
        worker = FlowerWorker(
            worker_id=args.worker_id,
            data_partition=args.data_partition,
            model_type=args.model_type,
            epochs_per_round=args.epochs_per_round,
            batch_size=args.batch_size,
            total_rounds=args.total_rounds
        )
        
        # Start Flower client - this will block until training completes
        fl.client.start_numpy_client(
            server_address=args.server_address,
            client=worker
        )
        
        # After training completes, exit gracefully
        logger.info(f"[{args.worker_id}] ✅ Training completed. Worker shutting down...")
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.info(f"\n[{args.worker_id}] Worker stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"[{args.worker_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

