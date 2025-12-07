# MLflow vs fl-dashboard: Complementary Tools

## Overview

Both **MLflow** and **fl-dashboard** are used together in this FL system. They serve different purposes and complement each other.

## Comparison Table

| Feature | fl-dashboard | MLflow |
|---------|-------------|--------|
| **Purpose** | Real-time operational monitoring | Experiment tracking & model registry |
| **When Used** | During training (live) | During + After training (historical) |
| **Update Frequency** | Real-time (every round) | Per round (logged) |
| **Data Retention** | Current session only | Permanent (all experiments) |
| **Use Case** | Monitor training progress | Compare experiments, manage models |
| **UI Type** | Live dashboard | Experiment management UI |
| **Model Registry** | ❌ No | ✅ Yes |
| **Experiment Comparison** | ❌ No | ✅ Yes |
| **Reproducibility** | ❌ No | ✅ Yes (parameters logged) |

## How They Work Together

### During Training

```
FL Server (flower_server_metrics.py)
    │
    ├───► fl-dashboard (Real-time monitoring)
    │     ├── Current round status
    │     ├── Live accuracy/loss charts
    │     ├── Network traffic visualization
    │     └── Worker status
    │
    └───► MLflow (Background logging)
          ├── Log metrics to database
          ├── Store model artifacts
          └── Track experiment parameters
```

### Code Flow

In `flower_server_metrics.py`, both are called:

```python
# 1. Send to fl-dashboard (real-time)
self._send_to_dashboard(dashboard_data)  # Line ~536

# 2. Log to MLflow (historical tracking)
if self.mlflow_enabled:
    mlflow.log_metrics({
        "round_accuracy": float(accuracy),
        "round_loss": float(loss),
        # ... more metrics
    }, step=rnd)  # Line ~550
```

## Use Cases

### Use fl-dashboard when:
- ✅ **Monitoring active training** - See what's happening right now
- ✅ **Debugging issues** - Real-time worker status
- ✅ **Quick checks** - Is training progressing?
- ✅ **Live visualization** - Watch charts update in real-time

### Use MLflow when:
- ✅ **Comparing experiments** - Which run performed better?
- ✅ **Model selection** - Which model version to deploy?
- ✅ **Reproducibility** - What parameters were used?
- ✅ **Historical analysis** - How did accuracy change over time?
- ✅ **Model registry** - Manage model versions and stages

## Example Workflow

### Scenario: Training a new model

1. **Start Training:**
   ```bash
   docker compose up -d
   ```

2. **Monitor in fl-dashboard** (http://localhost:5000):
   - Watch Round 1/5 → 2/5 → 3/5...
   - See accuracy improving: 60% → 75% → 85%
   - Check network traffic per round
   - Verify all workers are active

3. **After Training:**
   - Open MLflow UI (http://localhost:5000 - MLflow)
   - Compare this run with previous runs
   - See which parameters worked best
   - Register model for production

4. **Next Training Session:**
   - Try different parameters (e.g., more rounds, different DP settings)
   - Monitor in fl-dashboard (real-time)
   - Compare results in MLflow (historical)

## Data Flow

```
┌─────────────────────────────────────────┐
│   FL Training Session                   │
│   (flower_server_metrics.py)            │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴───────┐
       │               │
       ▼               ▼
┌─────────────┐  ┌──────────────┐
│ fl-dashboard│  │   MLflow     │
│             │  │              │
│ Real-time   │  │ Historical   │
│ Monitoring  │  │ Tracking     │
│             │  │              │
│ • Live UI   │  │ • Database    │
│ • Charts    │  │ • Artifacts   │
│ • Status    │  │ • Registry    │
└─────────────┘  └──────────────┘
```

## Port Configuration

Looking at `docker-compose.yml`:

- **fl-dashboard**: 
  - Host port: `5001` → Container port: `5000`
  - Access: `http://localhost:5001`
  - Internal Docker: `http://fl-dashboard:5000`

- **MLflow**: 
  - Host port: `5002` → Container port: `5000`
  - Access: `http://localhost:5002`
  - Internal Docker: `http://mlflow-server:5000`

**No port conflict!** They're on different host ports:
- ✅ **fl-dashboard**: `http://localhost:5001` (Real-time monitoring)
- ✅ **MLflow**: `http://localhost:5002` (Experiment tracking)

## Summary

**Both tools are used together:**
- ✅ **fl-dashboard** = Your "live TV" - watch training happen
- ✅ **MLflow** = Your "lab notebook" - record everything for later

**You don't choose one or the other - you use both!**

- During training: Watch fl-dashboard for real-time updates
- After training: Use MLflow to analyze, compare, and manage models

This gives you:
1. **Real-time visibility** (fl-dashboard)
2. **Historical tracking** (MLflow)
3. **Model management** (MLflow)
4. **Experiment comparison** (MLflow)

Both are essential for a complete MLOps setup! 🚀

