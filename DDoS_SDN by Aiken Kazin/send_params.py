import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from tensorflow.keras.models import load_model
import flwr as fl
from flwr.common import Weights, weights_to_parameters, parameters_to_weights


# Configuration
FL_SERVER_ADDRESS = os.getenv("FL_SERVER_ADDRESS", "localhost:8080")
DEFAULT_MODEL_DIR = "models"


def extract_model_weights(model_path: str) -> Weights:
    """
    Extract model weights for Federated Learning.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Flower Weights object containing model weights
    """
    print(f" Loading model from {model_path}...")
    model = load_model(model_path)
    
    # Extract weights as numpy arrays (Flower format)
    weights = []
    for layer in model.layers:
        layer_weights = layer.get_weights()
        if layer_weights:
            weights.extend([w for w in layer_weights])
    
    return weights


def load_training_metrics(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Load training metrics from the training report Excel file.
    
    Args:
        model_name: Name of the model (e.g., "CNN_LSTM.h5")
        
    Returns:
        Dictionary with training metrics or None if not found
    """
    model_base_name = model_name.replace(".h5", "")
    report_path = os.path.join("report", f"{model_base_name}_report.xlsx")
    
    if not os.path.exists(report_path):
        print(f" Warning: Training report not found at {report_path}")
        return None
    
    try:
        # Read the Summary sheet
        summary_df = pd.read_excel(report_path, sheet_name="Summary")
        metrics = {}
        
        for _, row in summary_df.iterrows():
            metric_name = row["Metric"]
            value = row["Value"]
            metrics[metric_name.lower().replace(" ", "_")] = float(value)
        
        # Read the History sheet for full training history
        try:
            history_df = pd.read_excel(report_path, sheet_name="History")
            metrics["training_history"] = {
                "epochs": int(history_df["epoch"].max()),
                "final_train_accuracy": float(history_df["accuracy"].iloc[-1]),
                "final_val_accuracy": float(history_df["val_accuracy"].iloc[-1]),
                "final_train_loss": float(history_df["loss"].iloc[-1]),
                "final_val_loss": float(history_df["val_loss"].iloc[-1]),
                "best_val_accuracy": float(history_df["val_accuracy"].max()),
                "best_val_loss": float(history_df["val_loss"].min())
            }
        except Exception as e:
            print(f" Warning: Could not read training history: {e}")
        
        return metrics
    except Exception as e:
        print(f" Error loading training metrics: {e}")
        return None


def load_evaluation_metrics(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Load evaluation metrics from the evaluation report Excel file.
    
    Args:
        model_name: Name of the model (e.g., "CNN_LSTM.h5")
        
    Returns:
        Dictionary with evaluation metrics or None if not found
    """
    # Find the latest evaluation report for this model
    report_dir = "report"
    if not os.path.exists(report_dir):
        return None
    
    # Look for evaluation reports matching the model name
    eval_reports = [
        f for f in os.listdir(report_dir)
        if f.startswith("evaluation_report_") and model_name in f and f.endswith(".xlsx")
    ]
    
    if not eval_reports:
        print(f" Warning: No evaluation report found for {model_name}")
        return None
    
    # Get the latest report
    latest_report = max(
        [os.path.join(report_dir, f) for f in eval_reports],
        key=os.path.getmtime
    )
    
    try:
        summary_df = pd.read_excel(latest_report, sheet_name="Summary")
        metrics = {}
        
        for _, row in summary_df.iterrows():
            metric_name = row["Metric"]
            value = row["Value"]
            metrics[metric_name.lower().replace("-", "_")] = float(value)
        
        # Try to get class-level metrics
        try:
            class_df = pd.read_excel(latest_report, sheet_name="Class_Report")
            metrics["class_metrics"] = class_df.to_dict(orient="records")
        except:
            pass
        
        return metrics
    except Exception as e:
        print(f" Error loading evaluation metrics: {e}")
        return None


def get_dataset_info() -> Dict[str, Any]:
    """
    Get information about the dataset used for training.
    
    Returns:
        Dictionary with dataset metadata
    """
    dataset_path = "dataset_sdn.csv"
    if not os.path.exists(dataset_path):
        return {}
    
    try:
        df = pd.read_csv(dataset_path)
        return {
            "num_samples": int(df.shape[0]),
            "num_features": int(df.shape[1]),
            "feature_names": df.columns.tolist()
        }
    except Exception as e:
        print(f" Warning: Could not load dataset info: {e}")
        return {}


class FlowerClient(fl.client.NumPyClient):
    """Flower client for sending model weights to the server"""
    
    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name
        self.model = load_model(model_path)
        
        # Load metadata
        self.training_metrics = load_training_metrics(model_name)
        self.eval_metrics = load_evaluation_metrics(model_name)
        self.dataset_info = get_dataset_info()
    
    def get_parameters(self, config: Dict[str, Any]) -> Weights:
        """Return model parameters (weights)"""
        return extract_model_weights(self.model_path)
    
    def get_properties(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Return client properties (metadata)"""
        return {
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "training_metrics": self.training_metrics or {},
            "evaluation_metrics": self.eval_metrics or {},
            "dataset_info": self.dataset_info,
            "model_config": {
                "input_shape": self.model.input_shape,
                "output_shape": self.model.output_shape,
                "num_layers": len(self.model.layers),
                "total_params": self.model.count_params(),
                "layer_names": [layer.name for layer in self.model.layers],
                "layer_types": [type(layer).__name__ for layer in self.model.layers]
            }
        }
    
    def fit(self, parameters: Weights, config: Dict[str, Any]) -> Tuple[Weights, int, Dict[str, Any]]:
        """Fit is called during federated training - we'll just return our weights"""
        weights = extract_model_weights(self.model_path)
        return weights, len(self.dataset_info.get("num_samples", 0)), {}
    
    def evaluate(self, parameters: Weights, config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate is called during federated evaluation"""
        # Return evaluation metrics if available
        metrics = self.eval_metrics or {}
        loss = metrics.get("test_loss", 0.0)
        num_samples = self.dataset_info.get("num_samples", 0)
        return float(loss), num_samples, metrics


def main():
    """Main function to send model weights to Flower server"""
    # Get model name from command line or detect latest
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        # Auto-detect latest model
        models_dir = DEFAULT_MODEL_DIR
        if not os.path.exists(models_dir):
            print("❌ Models directory not found")
            sys.exit(1)
        
        h5_files = [f for f in os.listdir(models_dir) if f.endswith(".h5")]
        if not h5_files:
            print("❌ No .h5 models found")
            sys.exit(1)
        
        latest_file = max(
            [os.path.join(models_dir, f) for f in h5_files],
            key=os.path.getmtime
        )
        model_name = os.path.basename(latest_file)
    
    model_path = os.path.join(DEFAULT_MODEL_DIR, model_name)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)
    
    # Get server address from environment
    server_address = os.getenv("FL_SERVER_ADDRESS", FL_SERVER_ADDRESS)
    
    # Parse server address (format: host:port)
    if ":" in server_address:
        server_host, server_port = server_address.split(":")
        server_port = int(server_port)
    else:
        server_host = server_address
        server_port = 8080
    
    print(f"\n{'='*60}")
    print(f" Federated Learning Client (Flower)")
    print(f"{'='*60}")
    print(f" Model: {model_name}")
    print(f" Server: {server_host}:{server_port}")
    print(f"{'='*60}\n")
    
    # Prepare client
    try:
        print(" Preparing client...")
        client = FlowerClient(model_path, model_name)
        
        print(" Connecting to Flower server...")
        print(" Sending model weights and metadata...\n")
        
        # Start Flower client
        fl.client.start_numpy_client(
            server_address=f"{server_host}:{server_port}",
            client=client
        )
        
        print(f"\n{'='*60}")
        print("✅ Model weights sent successfully to Flower server!")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n❌ Error connecting to Flower server: {e}")
        print("\nMake sure the Flower server is running:")
        print(f"  flwr server --address {server_host} --port {server_port}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
