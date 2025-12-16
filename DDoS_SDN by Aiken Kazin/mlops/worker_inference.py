"""
Worker-Side Inference API

Provides local inference API on workers for federated inference.
Data stays on worker, no need to send to server.
"""

import os
import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import Model

from mlops.worker_model_manager import WorkerModelManager
from preprocess import process_col, normalization

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global model manager
model_manager: Optional[WorkerModelManager] = None


def init_model_manager(worker_id: str = "worker1", models_dir: str = "/app/models"):
    """Initialize model manager"""
    global model_manager
    model_manager = WorkerModelManager(models_dir=models_dir, worker_id=worker_id)
    logger.info(f"✅ Worker Model Manager initialized for {worker_id}")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "worker_id": model_manager.worker_id if model_manager else "unknown",
        "models_loaded": list(model_manager.models_cache.keys()) if model_manager else []
    })


@app.route('/model/info', methods=['GET'])
def get_model_info():
    """Get information about loaded models"""
    if not model_manager:
        return jsonify({"error": "Model manager not initialized"}), 500
    
    model_type = request.args.get('model_type')
    if model_type:
        info = model_manager.get_model_info(model_type)
        return jsonify(info)
    else:
        # Return info for all loaded models
        all_info = {}
        for model_type in model_manager.models_cache.keys():
            model_type_clean = model_type.split('_')[0]  # Extract model type from cache key
            all_info[model_type_clean] = model_manager.get_model_info(model_type_clean)
        return jsonify(all_info)


@app.route('/model/load', methods=['POST'])
def load_model():
    """Load a model for inference"""
    if not model_manager:
        return jsonify({"error": "Model manager not initialized"}), 500
    
    data = request.json
    model_type = data.get('model_type')
    model_path = data.get('model_path')
    version = data.get('version', 'latest')
    
    if not model_type:
        return jsonify({"error": "model_type required"}), 400
    
    model = model_manager.load_model(
        model_type=model_type,
        model_path=model_path,
        version=version
    )
    
    if model:
        return jsonify({
            "success": True,
            "model_type": model_type,
            "version": version,
            "message": f"Model {model_type} loaded successfully"
        })
    else:
        return jsonify({
            "success": False,
            "error": f"Failed to load model {model_type}"
        }), 500


@app.route('/infer', methods=['POST'])
def infer():
    """
    Perform inference on single sample.
    
    Request body:
    {
        "features": [list of feature values],
        "model_type": "MLPv2" | "LSTM" | "CNN1D" | "CNN_LSTM"
    }
    
    Returns:
    {
        "prediction": 0 or 1,
        "probability": float,
        "probabilities": [prob_class_0, prob_class_1],
        "model_type": "MLPv2",
        "model_version": "latest"
    }
    """
    if not model_manager:
        return jsonify({"error": "Model manager not initialized"}), 500
    
    data = request.json
    features = data.get('features')
    model_type = data.get('model_type', 'MLPv2')
    
    if not features:
        return jsonify({"error": "features required"}), 400
    
    # Load model if not cached
    model = model_manager.load_model(model_type)
    if not model:
        return jsonify({"error": f"Model {model_type} not available"}), 404
    
    try:
        # Prepare input
        features_array = np.array([features])
        
        # Reshape for sequence models (LSTM, CNN1D, CNN_LSTM)
        if model_type in ['LSTM', 'CNN1D', 'CNN_LSTM']:
            # Add sequence dimension: (batch, timesteps, features)
            features_array = np.expand_dims(features_array, axis=2)
        
        # Predict
        predictions = model.predict(features_array, verbose=0)
        prediction_proba = predictions[0]
        
        # Get prediction
        predicted_class = int(np.argmax(prediction_proba))
        probability = float(prediction_proba[predicted_class])
        
        return jsonify({
            "prediction": predicted_class,
            "probability": probability,
            "probabilities": prediction_proba.tolist(),
            "model_type": model_type,
            "model_version": model_manager.model_versions.get(model_type, "latest")
        })
        
    except Exception as e:
        logger.error(f"❌ Inference error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/infer/batch', methods=['POST'])
def infer_batch():
    """
    Perform batch inference.
    
    Request body:
    {
        "features": [[sample1_features], [sample2_features], ...],
        "model_type": "MLPv2" | "LSTM" | "CNN1D" | "CNN_LSTM"
    }
    
    Returns:
    {
        "predictions": [0, 1, 0, ...],
        "probabilities": [[prob0, prob1], ...],
        "model_type": "MLPv2",
        "count": 10
    }
    """
    if not model_manager:
        return jsonify({"error": "Model manager not initialized"}), 500
    
    data = request.json
    features_list = data.get('features')
    model_type = data.get('model_type', 'MLPv2')
    
    if not features_list:
        return jsonify({"error": "features required"}), 400
    
    # Load model if not cached
    model = model_manager.load_model(model_type)
    if not model:
        return jsonify({"error": f"Model {model_type} not available"}), 404
    
    try:
        # Prepare input
        features_array = np.array(features_list)
        
        # Reshape for sequence models
        if model_type in ['LSTM', 'CNN1D', 'CNN_LSTM']:
            features_array = np.expand_dims(features_array, axis=2)
        
        # Predict
        predictions = model.predict(features_array, verbose=0)
        
        # Get predictions
        predicted_classes = [int(np.argmax(pred)) for pred in predictions]
        probabilities = predictions.tolist()
        
        return jsonify({
            "predictions": predicted_classes,
            "probabilities": probabilities,
            "model_type": model_type,
            "model_version": model_manager.model_versions.get(model_type, "latest"),
            "count": len(predicted_classes)
        })
        
    except Exception as e:
        logger.error(f"❌ Batch inference error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/model/update', methods=['POST'])
def update_model():
    """Receive model update notification from server"""
    if not model_manager:
        return jsonify({"error": "Model manager not initialized"}), 500
    
    data = request.json
    model_type = data.get('model_type')
    model_path = data.get('model_path')
    version = data.get('version', 'latest')
    
    if not model_type:
        return jsonify({"error": "model_type required"}), 400
    
    # Clear cache for this model type
    model_manager.clear_cache(model_type)
    
    # Load new model
    model = model_manager.load_model(
        model_type=model_type,
        model_path=model_path,
        version=version
    )
    
    if model:
        return jsonify({
            "success": True,
            "message": f"Model {model_type} updated to version {version}"
        })
    else:
        return jsonify({
            "success": False,
            "error": f"Failed to update model {model_type}"
        }), 500


def run_inference_server(
    host: str = "0.0.0.0",
    port: int = 6000,
    worker_id: str = "worker1",
    models_dir: str = "/app/models"
):
    """
    Run inference API server.
    
    Args:
        host: Host address
        port: Port number
        worker_id: Worker identifier
        models_dir: Directory containing models
    """
    init_model_manager(worker_id=worker_id, models_dir=models_dir)
    
    logger.info(f"🚀 Starting Worker Inference API on {host}:{port}")
    logger.info(f"   Worker ID: {worker_id}")
    logger.info(f"   Models directory: {models_dir}")
    
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Worker Inference API')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=6000, help='Port number')
    parser.add_argument('--worker-id', type=str, default='worker1', help='Worker ID')
    parser.add_argument('--models-dir', type=str, default='/app/models', help='Models directory')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s] - %(message)s'
    )
    
    run_inference_server(
        host=args.host,
        port=args.port,
        worker_id=args.worker_id,
        models_dir=args.models_dir
    )

