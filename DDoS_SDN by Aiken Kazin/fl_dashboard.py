#!/usr/bin/env python3
"""
Flower Federated Learning Dashboard - Multi-Model Support

Real-time visualization dashboard for monitoring FL training across multiple models:
- Per-model accuracy and loss
- Per-model network traffic
- Per-model parameters count
- Per-model round progress
- Overall statistics

Usage:
    python fl_dashboard.py --port 5000
"""

import os
import json
import time
import argparse
from datetime import datetime
from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
import threading
from collections import defaultdict

app = Flask(__name__)
CORS(app)

# Global metrics storage - per model type
# Structure: metrics[model_type] = {rounds, accuracies, losses, network, etc.}
metrics = defaultdict(lambda: {
    "rounds": [],
    "workers": {},
    "network": {
        "total_bytes_sent": 0,
        "total_bytes_received": 0,
        "round_times": [],
        "bytes_sent_per_round": [],
        "bytes_received_per_round": []
    },
    "aggregation": {
        "accuracies": [],
        "losses": []
    },
    "params_sent_per_round": [],
    "params_received_per_round": [],
    "model_params_count": 0,  # Total parameters in the model
    "current_round": 0,
    "total_rounds": 5
})

# Dashboard HTML template with per-model sections
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Flower FL Dashboard - Multi-Model</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .model-section {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border-left: 5px solid #667eea;
        }
        .model-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }
        .model-title {
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }
        .model-status {
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }
        .status-training { background-color: #fff3cd; color: #856404; }
        .status-complete { background-color: #d4edda; color: #155724; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-bottom: 15px;
        }
        .stat-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }
        .stat-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            color: #666;
            font-size: 0.85em;
            margin-top: 5px;
        }
        .chart-container {
            margin-top: 15px;
        }
        .chart-title {
            font-size: 1em;
            font-weight: bold;
            margin-bottom: 10px;
            color: #555;
        }
        .overview-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .overview-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .overview-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
        }
        .overview-label {
            color: #666;
            margin-top: 5px;
            font-size: 1.1em;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌺 Flower Federated Learning Dashboard</h1>
        <p>Multi-Model Training Monitor - Real-time tracking of all models</p>
        <p>Last updated: <span id="lastUpdate">-</span></p>
    </div>

    <div class="overview-grid">
        <div class="overview-card">
            <div class="overview-value" id="totalModels">0</div>
            <div class="overview-label">Active Models</div>
        </div>
        <div class="overview-card">
            <div class="overview-value" id="totalWorkers">0</div>
            <div class="overview-label">Active Workers</div>
        </div>
        <div class="overview-card">
            <div class="overview-value" id="totalRounds">0</div>
            <div class="overview-label">Total Rounds Completed</div>
        </div>
        <div class="overview-card">
            <div class="overview-value" id="totalBytes">0 MB</div>
            <div class="overview-label">Total Network Traffic</div>
        </div>
    </div>

    <div id="modelSections"></div>

    <script>
        function formatBytes(bytes) {
            if (bytes === 0) return '0 B';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
        }

        function formatNumber(num) {
            if (num >= 1000000) return (num / 1000000).toFixed(2) + 'M';
            if (num >= 1000) return (num / 1000).toFixed(2) + 'K';
            return num.toString();
        }

        function updateDashboard(data) {
            document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
            
            // Update overview
            const modelTypes = Object.keys(data.models || {});
            document.getElementById('totalModels').textContent = modelTypes.length;
            
            let totalWorkers = 0;
            let totalRounds = 0;
            let totalBytes = 0;
            
            modelTypes.forEach(modelType => {
                const modelData = data.models[modelType];
                totalWorkers += Object.keys(modelData.workers || {}).length;
                totalRounds += modelData.rounds.length;
                totalBytes += modelData.network.total_bytes_sent + modelData.network.total_bytes_received;
            });
            
            document.getElementById('totalWorkers').textContent = totalWorkers;
            document.getElementById('totalRounds').textContent = totalRounds;
            document.getElementById('totalBytes').textContent = formatBytes(totalBytes);
            
            // Update per-model sections
            let modelSectionsHtml = '';
            
            modelTypes.forEach(modelType => {
                const modelData = data.models[modelType];
                const currentRound = modelData.current_round || 0;
                const totalRounds = modelData.total_rounds || 5;
                const isComplete = currentRound >= totalRounds;
                const statusClass = isComplete ? 'status-complete' : 'status-training';
                const statusText = isComplete ? '✅ Complete' : '🔄 Training';
                
                const currentAccuracy = modelData.aggregation.accuracies.length > 0 
                    ? (modelData.aggregation.accuracies[modelData.aggregation.accuracies.length - 1] * 100).toFixed(2) + '%'
                    : 'N/A';
                const currentLoss = modelData.aggregation.losses.length > 0
                    ? modelData.aggregation.losses[modelData.aggregation.losses.length - 1].toFixed(4)
                    : 'N/A';
                
                const modelParams = modelData.model_params_count || 0;
                const bytesSent = formatBytes(modelData.network.total_bytes_sent);
                const bytesReceived = formatBytes(modelData.network.total_bytes_received);
                const totalParamsSent = (modelData.params_sent_per_round || []).reduce((a, b) => a + b, 0);
                const totalParamsReceived = (modelData.params_received_per_round || []).reduce((a, b) => a + b, 0);
                
                modelSectionsHtml += `
                    <div class="model-section">
                        <div class="model-header">
                            <div class="model-title">${modelType}</div>
                            <div class="model-status ${statusClass}">${statusText} - Round ${currentRound}/${totalRounds}</div>
                        </div>
                        
                        <div class="stats-grid">
                            <div class="stat-card">
                                <div class="stat-value">${currentAccuracy}</div>
                                <div class="stat-label">Current Accuracy</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">${currentLoss}</div>
                                <div class="stat-label">Current Loss</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">${formatNumber(modelParams)}</div>
                                <div class="stat-label">Model Parameters</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">${bytesSent}</div>
                                <div class="stat-label">Bytes Sent</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">${bytesReceived}</div>
                                <div class="stat-label">Bytes Received</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">${Object.keys(modelData.workers || {}).length}</div>
                                <div class="stat-label">Workers</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">${formatNumber(totalParamsSent)}</div>
                                <div class="stat-label">Total Params Sent</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">${formatNumber(totalParamsReceived)}</div>
                                <div class="stat-label">Total Params Received</div>
                            </div>
                        </div>
                        
                        <div class="chart-container">
                            <div class="chart-title">Accuracy & Loss Over Rounds</div>
                            <div id="chart-${modelType}-accuracy"></div>
                        </div>
                        
                        <div class="chart-container">
                            <div class="chart-title">Network Traffic Per Round</div>
                            <div id="chart-${modelType}-network"></div>
                        </div>
                        
                        <div class="chart-container">
                            <div class="chart-title">Parameters Sent/Received Per Round</div>
                            <div id="chart-${modelType}-params"></div>
                        </div>
                    </div>
                `;
            });
            
            document.getElementById('modelSections').innerHTML = modelSectionsHtml || '<p>No models training yet...</p>';
            
            // Render charts for each model
            modelTypes.forEach(modelType => {
                const modelData = data.models[modelType];
                const rounds = modelData.rounds || [];
                const accuracies = modelData.aggregation.accuracies || [];
                const losses = modelData.aggregation.losses || [];
                const bytesSent = modelData.network.bytes_sent_per_round || [];
                const bytesReceived = modelData.network.bytes_received_per_round || [];
                const paramsSent = modelData.params_sent_per_round || [];
                const paramsReceived = modelData.params_received_per_round || [];
                
                // Accuracy/Loss chart
                Plotly.newPlot(`chart-${modelType}-accuracy`, [
                    {
                        x: rounds,
                        y: accuracies.map(a => a * 100),
                        name: 'Accuracy',
                        type: 'scatter',
                        mode: 'lines+markers',
                        line: { color: '#4caf50', width: 2 },
                        marker: { size: 8 }
                    },
                    {
                        x: rounds,
                        y: losses,
                        name: 'Loss',
                        yaxis: 'y2',
                        type: 'scatter',
                        mode: 'lines+markers',
                        line: { color: '#f44336', width: 2 },
                        marker: { size: 8 }
                    }
                ], {
                    xaxis: { title: 'Round' },
                    yaxis: { title: 'Accuracy (%)', side: 'left' },
                    yaxis2: { title: 'Loss', overlaying: 'y', side: 'right' },
                    hovermode: 'x unified',
                    margin: { l: 50, r: 50, t: 20, b: 50 },
                    height: 300
                });
                
                // Network traffic chart
                Plotly.newPlot(`chart-${modelType}-network`, [
                    {
                        x: rounds,
                        y: bytesSent.map(b => b / 1024 / 1024),
                        name: 'Bytes Sent',
                        type: 'bar',
                        marker: { color: '#2196f3' }
                    },
                    {
                        x: rounds,
                        y: bytesReceived.map(b => b / 1024 / 1024),
                        name: 'Bytes Received',
                        type: 'bar',
                        marker: { color: '#ff9800' }
                    }
                ], {
                    xaxis: { title: 'Round' },
                    yaxis: { title: 'MB' },
                    barmode: 'group',
                    hovermode: 'x unified',
                    height: 250
                });
                
                // Parameters sent/received chart
                Plotly.newPlot(`chart-${modelType}-params`, [
                    {
                        x: rounds,
                        y: paramsSent,
                        name: 'Params Sent',
                        type: 'bar',
                        marker: { color: '#00bcd4' },
                        text: paramsSent.map(p => p.toLocaleString()),
                        textposition: 'outside',
                        hovertemplate: '<b>Round %{x}</b><br>Params Sent: %{text}<extra></extra>'
                    },
                    {
                        x: rounds,
                        y: paramsReceived,
                        name: 'Params Received',
                        type: 'bar',
                        marker: { color: '#ff5722' },
                        text: paramsReceived.map(p => p.toLocaleString()),
                        textposition: 'outside',
                        hovertemplate: '<b>Round %{x}</b><br>Params Received: %{text}<extra></extra>'
                    }
                ], {
                    xaxis: { title: 'Round' },
                    yaxis: { title: 'Number of Parameters', type: 'log' },
                    barmode: 'group',
                    hovermode: 'x unified',
                    height: 250
                });
            });
        }

        function fetchMetrics() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => updateDashboard(data))
                .catch(error => console.error('Error fetching metrics:', error));
        }

        // Initial load
        fetchMetrics();

        // Update every 2 seconds
        setInterval(fetchMetrics, 2000);
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/metrics')
def get_metrics():
    """Get current FL metrics for all models"""
    # Convert defaultdict to regular dict for JSON serialization
    models_dict = {}
    for model_type, model_metrics in metrics.items():
        # Calculate current round
        current_round = len(model_metrics["rounds"])
        
        models_dict[model_type] = {
            "current_round": current_round,
            "total_rounds": model_metrics.get("total_rounds", 5),
            "rounds": model_metrics["rounds"],
            "network": {
                "total_bytes_sent": model_metrics["network"]["total_bytes_sent"],
                "total_bytes_received": model_metrics["network"]["total_bytes_received"],
                "round_times": model_metrics["network"]["round_times"],
                "bytes_sent_per_round": model_metrics["network"].get("bytes_sent_per_round", []),
                "bytes_received_per_round": model_metrics["network"].get("bytes_received_per_round", [])
            },
            "aggregation": model_metrics["aggregation"],
            "workers": model_metrics["workers"],
            "params_sent_per_round": model_metrics.get("params_sent_per_round", []),
            "params_received_per_round": model_metrics.get("params_received_per_round", []),
            "model_params_count": model_metrics.get("model_params_count", 0)
        }
    
    response = {
        "models": models_dict
    }
    
    return jsonify(response)

@app.route('/api/update', methods=['POST'])
def update_metrics():
    """Update metrics (called by FL server/workers)"""
    data = request.json
    
    # Extract model type - try multiple fields
    model_type = data.get("model_type") or data.get("modelType") or "Unknown"
    
    # Initialize model metrics if not exists
    if model_type not in metrics:
        metrics[model_type] = {
            "rounds": [],
            "workers": {},
            "network": {
                "total_bytes_sent": 0,
                "total_bytes_received": 0,
                "round_times": [],
                "bytes_sent_per_round": [],
                "bytes_received_per_round": []
            },
            "aggregation": {
                "accuracies": [],
                "losses": []
            },
            "params_sent_per_round": [],
            "params_received_per_round": [],
            "model_params_count": 0,
            "current_round": 0,
            "total_rounds": 5
        }
    
    model_metrics = metrics[model_type]
    
    # Update round
    if "round" in data:
        round_num = data["round"]
        if round_num not in model_metrics["rounds"]:
            model_metrics["rounds"].append(round_num)
            model_metrics["rounds"].sort()
        model_metrics["current_round"] = round_num
    
    # Update total rounds if provided
    if "total_rounds" in data:
        model_metrics["total_rounds"] = data["total_rounds"]
    
    # Update accuracy
    if "accuracy" in data:
        round_num = data.get("round", len(model_metrics["rounds"]))
        # Ensure accuracies list matches rounds
        while len(model_metrics["aggregation"]["accuracies"]) < len(model_metrics["rounds"]):
            model_metrics["aggregation"]["accuracies"].append(0.0)
        if round_num in model_metrics["rounds"]:
            idx = model_metrics["rounds"].index(round_num)
            model_metrics["aggregation"]["accuracies"][idx] = float(data["accuracy"])
    
    # Update loss
    if "loss" in data:
        round_num = data.get("round", len(model_metrics["rounds"]))
        # Ensure losses list matches rounds
        while len(model_metrics["aggregation"]["losses"]) < len(model_metrics["rounds"]):
            model_metrics["aggregation"]["losses"].append(0.0)
        if round_num in model_metrics["rounds"]:
            idx = model_metrics["rounds"].index(round_num)
            model_metrics["aggregation"]["losses"][idx] = float(data["loss"])
    
    # Update bytes sent
    if "bytes_sent" in data:
        model_metrics["network"]["total_bytes_sent"] += data["bytes_sent"]
        if "round" in data:
            round_num = data["round"]
            if round_num not in model_metrics["rounds"]:
                model_metrics["rounds"].append(round_num)
                model_metrics["rounds"].sort()
            round_idx = model_metrics["rounds"].index(round_num)
            while len(model_metrics["network"]["bytes_sent_per_round"]) <= round_idx:
                model_metrics["network"]["bytes_sent_per_round"].append(0)
            model_metrics["network"]["bytes_sent_per_round"][round_idx] += data["bytes_sent"]
    
    # Update bytes received
    if "bytes_received" in data:
        model_metrics["network"]["total_bytes_received"] += data["bytes_received"]
        if "round" in data:
            round_num = data["round"]
            if round_num not in model_metrics["rounds"]:
                model_metrics["rounds"].append(round_num)
                model_metrics["rounds"].sort()
            round_idx = model_metrics["rounds"].index(round_num)
            while len(model_metrics["network"]["bytes_received_per_round"]) <= round_idx:
                model_metrics["network"]["bytes_received_per_round"].append(0)
            model_metrics["network"]["bytes_received_per_round"][round_idx] += data["bytes_received"]
    
    # Update round time
    if "round_time" in data:
        model_metrics["network"]["round_times"].append(data["round_time"])
    
    # Update worker info
    if "worker_id" in data:
        worker_id = data["worker_id"]
        if worker_id not in model_metrics["workers"]:
            model_metrics["workers"][worker_id] = {"rounds_participated": 0, "active": False}
        model_metrics["workers"][worker_id]["active"] = data.get("active", True)
        model_metrics["workers"][worker_id]["last_seen"] = datetime.now().isoformat()
        if "round" in data:
            model_metrics["workers"][worker_id]["rounds_participated"] += 1
    
    # Update params sent
    if "params_sent" in data:
        if "round" in data:
            round_num = data["round"]
            if round_num not in model_metrics["rounds"]:
                model_metrics["rounds"].append(round_num)
                model_metrics["rounds"].sort()
            round_idx = model_metrics["rounds"].index(round_num)
            while len(model_metrics["params_sent_per_round"]) <= round_idx:
                model_metrics["params_sent_per_round"].append(0)
            model_metrics["params_sent_per_round"][round_idx] += data["params_sent"]
    
    # Update params received
    if "params_received" in data:
        if "round" in data:
            round_num = data["round"]
            if round_num not in model_metrics["rounds"]:
                model_metrics["rounds"].append(round_num)
                model_metrics["rounds"].sort()
            round_idx = model_metrics["rounds"].index(round_num)
            while len(model_metrics["params_received_per_round"]) <= round_idx:
                model_metrics["params_received_per_round"].append(0)
            model_metrics["params_received_per_round"][round_idx] += data["params_received"]
    
    # Update model parameters count (from worker metrics)
    if "model_params_count" in data:
        model_metrics["model_params_count"] = data["model_params_count"]
    
    return jsonify({"status": "ok", "model_type": model_type})

def main():
    parser = argparse.ArgumentParser(description='Flower FL Dashboard - Multi-Model')
    parser.add_argument('--port', type=int, default=5000, help='Dashboard port (default: 5000)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address (default: 0.0.0.0)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print(" Flower Federated Learning Dashboard - Multi-Model")
    print("="*60)
    print(f" Dashboard URL: http://{args.host}:{args.port}")
    print(f" API Endpoint: http://{args.host}:{args.port}/api/metrics")
    print("="*60)
    print("\n Tracking metrics for all model types:")
    print("  • MLPv2")
    print("  • CNN1D")
    print("  • LSTM")
    print("  • CNN_LSTM")
    print("="*60 + "\n")
    
    app.run(host=args.host, port=args.port, debug=False, threaded=True)

if __name__ == "__main__":
    from flask import request
    main()
