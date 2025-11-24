# Federated Learning Integration with Flower

This directory includes scripts for sending model parameters to a Federated Learning server using [Flower](https://flower.dev/) (flwr) after training and evaluation.

## What is Flower?

Flower is a framework for building federated learning systems. It provides:
- **Server**: Coordinates federated learning rounds
- **Client**: Trains models locally and sends updates to the server
- **Built-in communication**: Handles all the networking and protocol details

## Quick Start

### Option A: Using Docker with Workers + Dashboard (Recommended for Federated Learning)

1. **Start the complete federated learning setup** (1 server + 2 workers + dashboard):
   ```bash
   docker-compose up -d
   ```
   
   This starts:
   - 1 Flower server (coordinates federated learning with metrics tracking)
   - 2 Workers/Agents (train models locally and participate in FL rounds)
   - 1 Dashboard (real-time visualization at http://localhost:5000)

2. **Open the Dashboard**:
   Open your browser and go to: **http://localhost:5000**
   
   The dashboard shows:
   - 📊 Current round and progress
   - 📈 Model accuracy and loss over rounds
   - 🌐 Network traffic (bytes sent/received)
   - ⏱️ Round duration (performance monitoring)
   - 📦 Parameters sent/received per round
   - 👥 Worker status and participation

3. **Auto-shutdown after training**:
   Containers are configured to **automatically stop** when training completes:
   - Server exits after all rounds finish
   - Workers exit when server disconnects
   - Dashboard stays running for viewing final results
   
   To manually stop everything:
   ```bash
   docker-compose down
   ```

4. **View logs**:
   ```bash
   docker-compose logs -f
   ```
   
   Or monitor specific service:
   ```bash
   docker-compose logs -f flower-server
   ```

### Option B: Using Docker (Server Only)

1. **Start the Flower Server**:
   ```bash
   ./docker-run.sh build
   ./docker-run.sh start
   ```

2. **Send Model Parameters** (from your host machine):
   ```bash
   ./train_evaluate.sh
   ```
   
   Or manually:
   ```bash
   python send_params.py CNN_LSTM.h5
   ```

### Option B: Local Installation

1. **Install Flower**:
   ```bash
   ./setup_flower.sh
   ```
   
   Or manually:
   ```bash
   pip install flwr
   ```

2. **Start the Flower Server**:
   ```bash
   python flower_server.py
   ```
   
   Or with custom settings:
   ```bash
   python flower_server.py --address 0.0.0.0 --port 8080 --min-clients 1
   ```

3. **Send Model Parameters**:
   ```bash
   ./train_evaluate.sh
   ```
   
   Or manually:
   ```bash
   python send_params.py CNN_LSTM.h5
   ```

## Setup

### Required Dependencies

Flower is already included in `requirement.txt`. Install it with:
```bash
pip install flwr
```

Or use the setup script:
```bash
./setup_flower.sh
```

## Configuration

### Environment Variables

You can configure the FL parameter sending via environment variables:

- `FL_SERVER_ADDRESS`: Flower server address in format `host:port` (default: `localhost:8080`)
- `SEND_FL_PARAMS`: Set to `"false"` to skip sending FL parameters (default: `"true"`)

### Example Usage

```bash
# Run with default settings (connects to localhost:8080)
./train_evaluate.sh

# Run with custom Flower server address
FL_SERVER_ADDRESS="192.168.1.50:8080" ./train_evaluate.sh

# Skip sending FL parameters
SEND_FL_PARAMS="false" ./train_evaluate.sh
```

## Running the Flower Server

Before sending model parameters, you need to start a Flower server.

### Using Docker (Recommended for Production)

Docker provides an isolated, reproducible environment for the Flower server.

#### Quick Start with Docker

```bash
# Build and start the server
./docker-run.sh build
./docker-run.sh start

# Or use docker-compose directly
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the server
docker-compose down
```

#### Docker Helper Script

We provide a convenient helper script (`docker-run.sh`) for common operations:

```bash
# Build the Docker image
./docker-run.sh build

# Start the server
./docker-run.sh start

# View logs
./docker-run.sh logs

# Check status
./docker-run.sh status

# Stop the server
./docker-run.sh stop

# Restart the server
./docker-run.sh restart

# Open shell in container
./docker-run.sh shell

# Clean up (removes container and image)
./docker-run.sh clean
```

#### Docker Compose Configuration

The `docker-compose.yml` file allows you to customize the server:

```bash
# Custom port
FL_PORT=9090 docker-compose up -d

# Custom minimum clients
FL_MIN_CLIENTS=3 docker-compose up -d

# All options
FL_PORT=9090 \
FL_MIN_CLIENTS=2 \
FL_NUM_ROUNDS=5 \
docker-compose up -d
```

#### Docker Benefits

- ✅ **Isolated environment**: No conflicts with local Python packages
- ✅ **Easy deployment**: Same setup works everywhere
- ✅ **Resource management**: Control CPU/memory usage
- ✅ **Health checks**: Automatic monitoring and restart
- ✅ **Portability**: Run on any machine with Docker

### Using the Custom Server Script (Local Development)

We provide a custom server script (`flower_server.py`) with better configuration options:

```bash
# Basic server (default: 0.0.0.0:8080)
python flower_server.py

# Custom address and port
python flower_server.py --address 0.0.0.0 --port 8080

# With more options
python flower_server.py \
    --address 192.168.1.50 \
    --port 8080 \
    --min-clients 1 \
    --num-rounds 1 \
    --fraction-fit 1.0 \
    --fraction-evaluate 1.0
```

### Using Flower CLI (Alternative)

You can also use Flower's built-in CLI:

```bash
# Basic server (default port 8080)
flwr server

# Custom address and port
flwr server --address 0.0.0.0 --port 8080

# With more options
flwr server --address 192.168.1.50 --port 8080 --grpc-max-message-length 536870912
```

The server will wait for clients to connect and send their model weights. It uses **Federated Averaging (FedAvg)** strategy by default.

## What Gets Sent

The `send_params.py` script sends the following Federated Learning parameters via Flower:

1. **Model Weights**: All layer weights in Flower's native format
2. **Model Architecture**: Input/output shapes, layer types, total parameters
3. **Training Metrics**: Final train/val accuracy and loss, best metrics, training history
4. **Evaluation Metrics**: Test accuracy, precision, recall, F1-score, class-level metrics
5. **Dataset Info**: Number of samples, features, feature names
6. **Training Configuration**: Epochs, batch size, optimizer settings, etc.

## Manual Execution

You can also run `send_params.py` manually:

```bash
# Send parameters for a specific model (connects to default server localhost:8080)
python send_params.py CNN_LSTM.h5

# With custom server address
FL_SERVER_ADDRESS="192.168.1.50:8080" python send_params.py CNN_LSTM.h5

# Auto-detect latest model
python send_params.py
```

## How It Works

1. **Training**: Your model is trained locally using `train.py`
2. **Evaluation**: Model is evaluated using `evaluate.py`
3. **Flower Client**: `send_params.py` creates a Flower client that:
   - Loads the trained model
   - Extracts model weights
   - Loads training/evaluation metrics
   - Connects to the Flower server
   - Sends weights and metadata to the server

## Flower Architecture

```
┌─────────────────┐
│  Flower Server  │  ← Coordinates federated learning
│  (flwr server)  │
└────────┬────────┘
         │
         │ gRPC
         │
┌────────▼────────┐
│  Flower Client  │  ← Your send_params.py script
│  (send_params)  │
└─────────────────┘
```

## Advantages of Flower

- ✅ **No custom API needed**: Flower handles all communication
- ✅ **Standard protocol**: Uses gRPC for efficient communication
- ✅ **Built-in features**: Aggregation, encryption, compression, etc.
- ✅ **Production-ready**: Used by many organizations
- ✅ **Easy to extend**: Can add custom strategies and clients

## Architecture

The federated learning setup consists of:

```
┌─────────────────┐
│  Flower Server  │  ← Coordinates federated learning rounds
│  (1 instance)    │     Aggregates weights from workers
└────────┬────────┘
         │
         │ gRPC
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌───▼───┐
│Worker1│ │Worker2│  ← Train models locally on their data
│Agent  │ │Agent  │     Send updated weights to server
└───────┘ └───────┘
```

## Visualization Dashboard

The FL dashboard provides real-time monitoring of federated learning training:

### Features

- **📊 Real-time Metrics**: Live updates every 2 seconds
- **📈 Model Performance**: Accuracy and loss over rounds
- **🌐 Network Traffic**: Bytes sent/received per round (identify slow cycles)
- **⏱️ Performance Monitoring**: Round duration to detect bottlenecks
- **📦 Parameter Tracking**: Number of parameters sent/received
- **👥 Worker Status**: Active workers and participation stats
- **🎯 Aggregation Results**: Latest model accuracy after aggregation

### Accessing the Dashboard

After starting with `docker-compose up -d`, open:
```
http://localhost:5000
```

The dashboard automatically collects metrics from the Flower server and displays:
- Current round progress
- Model accuracy/loss trends
- Network traffic visualization
- Round performance metrics
- Worker participation

## Files Overview

### Core Files
- **`flower_server.py`**: Basic Flower server script
- **`flower_server_metrics.py`**: Enhanced server with metrics tracking for dashboard
- **`flower_worker.py`**: Flower worker/agent that participates in federated learning
- **`fl_dashboard.py`**: Web dashboard for visualizing FL metrics
- **`send_params.py`**: Flower client that sends model weights to the server (one-time)
- **`setup_flower.sh`**: Setup script to install Flower and verify installation
- **`train_evaluate.sh`**: Training pipeline that optionally sends parameters via Flower

### Docker Files
- **`Dockerfile`**: Docker image definition for Flower server
- **`Dockerfile.worker`**: Docker image definition for Flower workers/agents
- **`Dockerfile.dashboard`**: Docker image definition for FL dashboard
- **`docker-compose.yml`**: Docker Compose configuration (1 server + 2 workers + dashboard)
- **`docker-run.sh`**: Helper script for Docker operations
- **`requirements-server.txt`**: Minimal dependencies for the server (only Flower)
- **`requirements-worker.txt`**: Full ML stack for workers (TensorFlow, etc.)
- **`.dockerignore`**: Files to exclude from Docker build

## Troubleshooting

### Connection Error
If you see connection errors, make sure:
1. Flower server is running:
   - Docker: `docker-compose ps` or `./docker-run.sh status`
   - Local: `python flower_server.py --address <host> --port <port>`
2. Server address matches `FL_SERVER_ADDRESS` environment variable
3. Firewall allows connections on the specified port
4. Both server and client are using the same address and port
5. If using Docker, ensure the port is correctly mapped (check `docker-compose.yml`)

### Docker Issues
- **Port already in use**: Change the port in `docker-compose.yml` or stop the conflicting service
- **Container won't start**: Check logs with `docker-compose logs`
- **Can't connect from host**: Ensure Docker port mapping is correct (host:container)
- **Permission denied**: Make sure Docker daemon is running and you have permissions

### Model Loading Error
Make sure:
1. Model file exists in `models/` directory
2. Model is a valid Keras `.h5` file
3. TensorFlow is installed correctly

### Missing Metrics
If training/evaluation metrics are missing:
- Check that `report/` directory contains the Excel reports
- Reports should be named `{model_name}_report.xlsx` and `evaluation_report_{model_name}_*.xlsx`

## Next Steps

To build a complete federated learning system:

1. **Server Strategy**: Implement aggregation strategies (FedAvg, FedProx, etc.)
2. **Multiple Clients**: Connect multiple clients for federated training
3. **Custom Logic**: Add custom fit/evaluate methods for your use case
4. **Monitoring**: Use Flower's built-in monitoring and logging

See [Flower Documentation](https://flower.dev/docs/) for more details.
