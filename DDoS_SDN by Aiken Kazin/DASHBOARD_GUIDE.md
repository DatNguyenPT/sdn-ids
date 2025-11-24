# FL Dashboard Quick Guide

## Overview

The FL Dashboard provides real-time visualization of federated learning training metrics.

## Access

After starting with `docker-compose up -d`, open:
```
http://localhost:5000
```

## Visualizations

### 1. **Network Traffic** 📊
- **Bytes Sent**: Total data sent from server to workers
- **Bytes Received**: Total data received from workers
- **Per Round**: See traffic per round to identify slow cycles
- **Location**: "Network Traffic" chart

### 2. **Parameters Sent/Received** 📦
- **Params Sent**: Number of parameters sent to workers each round
- **Params Received**: Number of parameters received from workers
- **Location**: "Parameters Sent/Received Per Round" chart

### 3. **Aggregation Results** 🎯
- **Latest Accuracy**: Current model accuracy after aggregation
- **Latest Loss**: Current model loss after aggregation
- **Trends**: See accuracy/loss improvement over rounds
- **Location**: "Model Accuracy & Loss Over Rounds" chart + stats cards

### 4. **Performance Monitoring** ⏱️
- **Round Duration**: Time taken for each FL round
- **Average Round Time**: Overall performance metric
- **Slow Cycle Detection**: Identify rounds that take longer
- **Location**: "Round Duration" chart + "Avg Round Time" stat

### 5. **Worker Status** 👥
- **Active Workers**: Number of workers currently participating
- **Worker Participation**: Rounds each worker has participated in
- **Last Seen**: When each worker was last active
- **Location**: "Worker Status" section

## Key Metrics Explained

- **Current Round**: Which FL round is currently running
- **Total Rounds**: Total number of rounds configured
- **Current Accuracy**: Latest aggregated model accuracy
- **Current Loss**: Latest aggregated model loss
- **Total Bytes Sent/Received**: Cumulative network traffic
- **Avg Round Time**: Average time per round (performance indicator)

## Using the Dashboard

1. **Start FL Training**: `docker-compose up -d`
2. **Open Dashboard**: http://localhost:5000
3. **Monitor Progress**: Dashboard updates every 2 seconds automatically
4. **Identify Issues**:
   - Slow rounds → Check "Round Duration" chart
   - High network usage → Check "Network Traffic" chart
   - Low accuracy → Check "Accuracy & Loss" chart
   - Worker issues → Check "Worker Status" section

## Troubleshooting

- **Dashboard not loading**: Check if `fl-dashboard` container is running
- **No metrics**: Ensure `flower-server` is using `flower_server_metrics.py`
- **Stale data**: Refresh the page or check server logs

