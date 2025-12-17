# MLOps for Federated Learning: Adapting Operations to Distributed Training

**How MLOps Principles Apply to Your DDoS Detection FL System**

## Introduction

Federated Learning (FL) introduces unique challenges that require adapted MLOps practices. Unlike centralized ML, FL involves:

- Distributed training across multiple clients
- Privacy constraints (no access to client data)
- Round-based iterative training
- Model aggregation instead of direct training
- Communication overhead management

## Key Differences: Centralized ML vs Federated Learning

| **Aspect** | **Centralized ML** | **Federated Learning** |
|------------|-------------------|------------------------|
| **Training** | Single location | Distributed across clients |
| **Data Access** | Full dataset available | Data stays on clients |
| **Model Updates** | Direct gradient updates | Weight aggregation (FedAvg) |
| **Monitoring** | Single training run | Multiple rounds + clients |
| **Versioning** | Model versions | Round versions + aggregated models |
| **Deployment** | Single model | Global model + client models |

## MLOps Components Adapted for FL

### 1. Experiment Tracking (FL-Specific)

#### What to Track in FL:

**Per-Round Metrics:**

- Global model accuracy/loss (aggregated)
- Client participation rate
- Round completion time
- Communication overhead (bytes sent/received)
- Convergence rate across rounds

**Per-Client Metrics (Privacy-Preserving):**

- Local training loss/accuracy (aggregated, not individual)
- Number of samples per client
- Training time per client
- Model update contribution (weighted by samples)

**In Your Code:**

```python
# flower_server_metrics.py tracks:
- Round number
- Aggregated accuracy/loss
- Bytes sent/received
- Parameters count
- Round duration
- Client participation
```

#### MLOps Tools for FL Experiment Tracking:

**MLflow Integration:**

```python
import mlflow

# Track FL experiment
with mlflow.start_run():
    mlflow.log_param("num_rounds", 5)
    mlflow.log_param("num_clients", 2)
    mlflow.log_param("fraction_fit", 1.0)
    
    # Track per-round metrics
    for round_num in range(1, 6):
        mlflow.log_metric("round_accuracy", accuracy, step=round_num)
        mlflow.log_metric("round_loss", loss, step=round_num)
        mlflow.log_metric("bytes_sent", bytes_sent, step=round_num)
        mlflow.log_metric("round_time", round_time, step=round_num)
    
    # Log final aggregated model
    mlflow.keras.log_model(aggregated_model, "model")
```

### 2. Model Versioning (Round-Based)

#### FL Model Versioning Strategy:

**Current Approach:**

- Single model saved: `MLPv2_FL.h5`
- No versioning of rounds
- No tracking of intermediate rounds

**MLOps Approach:**

$$\text{Model Version} = \text{Model\_Type}\_\text{Round}\_\text{Accuracy}\_\text{Timestamp}$$

**Example:**

- `MLPv2_FL_Round1_Acc0.85_20250115.h5`
- `MLPv2_FL_Round3_Acc0.91_20250115.h5`
- `MLPv2_FL_Round5_Acc0.94_20250115.h5` (final)

**Implementation:**

```python
# After each round aggregation
round_accuracy = aggregated_metrics.get('accuracy', 0)
model_version = f"{model_type}_FL_Round{round_num}_Acc{round_accuracy:.3f}_{timestamp}"
model.save(f"models/{model_version}.h5")

# Register in model registry
mlflow.register_model(
    f"runs:/{run_id}/model",
    f"{model_type}_FL_v{round_num}"
)
```

#### Why Round-Based Versioning Matters:

- **Rollback capability:** Can revert to previous round if performance degrades
- **Convergence analysis:** Track how model improves across rounds
- **Debugging:** Identify which round introduced issues
- **Research:** Compare different FL strategies (FedAvg vs FedProx vs others)

### 3. Model Registry (FL-Aware)

#### Registry Structure:

$$\text{Registry} = \begin{cases}
\text{Global Models} & \text{(Aggregated, server-side)} \\
\text{Client Models} & \text{(Local, client-side, optional)}
\end{cases}$$

**Global Model Registry:**

- **Entry:** Aggregated model after each round
- **Metadata:** Round number, accuracy, loss, client count, aggregation method
- **Staging:** Round-by-round progression
- **Production:** Final model after all rounds

**Example Registry Entry:**

```json
{
  "model_id": "MLPv2_FL_Round5",
  "version": "5.0",
  "round": 5,
  "accuracy": 0.94,
  "loss": 0.12,
  "num_clients": 2,
  "total_samples": 72235,
  "aggregation_method": "FedAvg",
  "timestamp": "2025-01-15T10:30:00",
  "artifacts": {
    "model_path": "models/MLPv2_FL_Round5.h5",
    "metrics_path": "metrics/round5_metrics.json"
  }
}
```

### 4. Monitoring (FL-Specific Metrics)

#### What to Monitor in FL:

**Training Metrics:**

- **Convergence:** How quickly global model converges
- **Stability:** Variance in client updates
- **Participation:** Client dropout rate, availability
- **Communication:** Network overhead, round time

**Model Quality Metrics:**

- **Global accuracy:** Aggregated model performance
- **Client accuracy variance:** How consistent are client models
- **Data distribution:** IID vs non-IID detection
- **Model drift:** Performance degradation over rounds

**In Your Code:**

```python
# flower_server_metrics.py already tracks:
- Round-by-round accuracy/loss
- Bytes sent/received per round
- Round duration
- Client participation
- Parameter counts
```

#### MLOps Monitoring Dashboard:

**Key Metrics to Display:**

$$\begin{aligned}
\text{Convergence Rate} &= \frac{\text{Accuracy}_{\text{round }n} - \text{Accuracy}_{\text{round }1}}{\text{round }n - 1} \\
\text{Communication Efficiency} &= \frac{\text{Model Improvement}}{\text{Bytes Transferred}} \\
\text{Client Contribution} &= \frac{n_k \cdot \text{Accuracy}_k}{\sum n_k}
\end{aligned}$$

### 5. Model Deployment (FL-Specific)

#### Deployment Architecture:

$$\text{FL Deployment} = \begin{cases}
\text{Global Model} & \text{(Server, aggregated)} \\
\text{Client Models} & \text{(Optional, local inference)}
\end{cases}$$

**Option 1: Centralized Deployment (Current)**

- Deploy aggregated global model on server
- Clients send inference requests to server
- **Pros:** Single model to maintain, consistent predictions
- **Cons:** Network latency, server load

**Option 2: Federated Deployment**

- Deploy global model to each client
- Clients perform local inference
- **Pros:** Low latency, privacy-preserving
- **Cons:** Model synchronization needed

**Option 3: Hybrid**

- Global model on server for centralized inference
- Client models for local inference (optional)
- **Pros:** Flexibility, fallback options

#### Deployment Pipeline:

```python
# After FL training completes
def deploy_fl_model(model_path, model_version):
    # 1. Load aggregated model
    global_model = load_model(model_path)
    
    # 2. Validate performance
    test_accuracy = evaluate_model(global_model)
    if test_accuracy < threshold:
        raise ValueError("Model performance below threshold")
    
    # 3. Register in model registry
    mlflow.register_model(model_path, f"MLPv2_FL_{model_version}")
    
    # 4. Deploy to production
    # Option A: Server deployment
    deploy_to_server(global_model)
    
    # Option B: Client deployment (federated)
    deploy_to_clients(global_model)
    
    # 5. Monitor deployment
    start_monitoring(global_model)
```

### 6. Automated Retraining (FL-Specific)

#### Retraining Triggers:

**1. Scheduled Retraining:**

- Weekly/Monthly FL rounds
- New data accumulates on clients
- Global model needs updates

**2. Performance Degradation:**

- Monitor production accuracy
- If accuracy drops below threshold → trigger FL retraining
- New attack patterns detected

**3. Data Drift Detection:**

- Detect distribution shift in client data
- New DDoS attack types emerge
- Network behavior changes

**4. New Clients Join:**

- New network nodes added
- Retrain to incorporate new client data
- Maintain model freshness

#### FL Retraining Pipeline:

```python
def fl_retraining_pipeline():
    # 1. Check if retraining needed
    if not should_retrain():
        return
    
    # 2. Load previous model as initialization
    previous_model = load_model("models/MLPv2_FL_latest.h5")
    
    # 3. Start FL training with warm start
    fl_server.start_training(
        initial_weights=previous_model.get_weights(),
        num_rounds=5
    )
    
    # 4. Monitor training
    monitor_fl_rounds()
    
    # 5. Compare with previous model
    new_accuracy = evaluate_new_model()
    old_accuracy = evaluate_old_model()
    
    # 6. Deploy if improved
    if new_accuracy > old_accuracy:
        deploy_new_model()
    else:
        keep_old_model()
```

## FL-Specific MLOps Challenges

### 1. Privacy Constraints

**Challenge:** Cannot access client data for monitoring/validation

**MLOps Solution:**

- **Federated metrics:** Aggregate metrics without exposing individual client data
- **Differential privacy:** Add noise to aggregated metrics
- **Secure aggregation:** Use cryptographic protocols
- **Metadata only:** Track only aggregated statistics

**In Your Code:**

```python
# Privacy-preserving metrics
aggregated_accuracy = weighted_average(client_accuracies, client_samples)
# No individual client data exposed
```

### 2. Communication Overhead

**Challenge:** Model weights are large, multiple rounds increase communication

**MLOps Solution:**

- **Compression:** Quantization, pruning, compression
- **Selective updates:** Only send significant weight changes
- **Monitoring:** Track bytes sent/received (you already do this!)
- **Optimization:** Reduce model size, use efficient formats

**In Your Code:**

```python
# Already tracking communication
bytes_sent = self._estimate_bytes(weights_bytes)
params_sent = self._count_parameters(weights)
# Could optimize: compression, quantization
```

### 3. Client Heterogeneity

**Challenge:** Different clients have different:

- Data distributions (IID vs non-IID)
- Data sizes
- Computational resources
- Network conditions

**MLOps Solution:**

- **Adaptive strategies:** FedProx, FedNova for heterogeneous clients
- **Client selection:** Select clients based on resources/data
- **Monitoring:** Track client participation, performance variance
- **Fairness:** Ensure all clients benefit from FL

### 4. Round-Based Training

**Challenge:** Training happens in rounds, not continuous

**MLOps Solution:**

- **Round tracking:** Version models per round
- **Convergence monitoring:** Track improvement across rounds
- **Early stopping:** Stop if no improvement
- **Checkpointing:** Save models after each round

## MLOps Implementation for Your FL System

### Current State

**What You Have:**

- ✅ Basic monitoring (Flower dashboard)
- ✅ Model saving (H5 files)
- ✅ Metrics tracking (accuracy, loss, bytes)
- ✅ Docker containerization
- ✅ Round-based training

**What's Missing:**

- ❌ Model versioning
- ❌ Experiment tracking (MLflow)
- ❌ Model registry
- ❌ Production deployment
- ❌ Automated retraining

### Recommended MLOps Stack for FL

#### Tier 1: Essential (Start Here)

**1. MLflow for FL Experiment Tracking**

```python
# Track FL experiments
import mlflow

mlflow.set_experiment("DDoS_Detection_FL")

with mlflow.start_run():
    # Log FL configuration
    mlflow.log_params({
        "num_rounds": 5,
        "num_clients": 2,
        "model_type": "MLPv2",
        "aggregation": "FedAvg"
    })
    
    # Track per-round metrics
    for round_num in range(1, 6):
        mlflow.log_metrics({
            "round_accuracy": accuracy[round_num],
            "round_loss": loss[round_num],
            "bytes_sent": bytes_sent[round_num],
            "round_time": round_time[round_num]
        }, step=round_num)
    
    # Log final model
    mlflow.keras.log_model(final_model, "model")
```

**2. Model Versioning**

```python
# Version models by round
def save_fl_model(model, round_num, accuracy, model_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = f"{model_type}_FL_Round{round_num}_Acc{accuracy:.3f}_{timestamp}"
    model_path = f"models/{version}.h5"
    model.save(model_path)
    
    # Also save as latest
    latest_path = f"models/{model_type}_FL_latest.h5"
    model.save(latest_path)
    
    return model_path
```

#### Tier 2: Production (Next Steps)

**3. Model Registry & Versioning**

- Automated FL training on schedule
- Model validation after each round
- Automated deployment if improved
- Rollback capability

**4. Production Monitoring**

- Monitor global model performance
- Track client participation
- Detect data drift
- Alert on performance degradation

#### Tier 3: Advanced

**5. Federated Model Serving**

- Deploy models to clients
- Local inference capabilities
- Model synchronization

**6. Advanced FL Strategies**

- FedProx (handles heterogeneity)
- Secure aggregation
- Differential privacy integration

## FL-Specific MLOps Metrics

### Key Performance Indicators (KPIs)

**Training Metrics:**

$$\begin{aligned}
\text{Convergence Rate} &= \frac{\Delta \text{Accuracy}}{\text{Rounds}} \\
\text{Communication Efficiency} &= \frac{\text{Accuracy Gain}}{\text{Total Bytes}} \\
\text{Client Participation Rate} &= \frac{\text{Active Clients}}{\text{Total Clients}} \\
\text{Round Efficiency} &= \frac{\text{Accuracy Improvement}}{\text{Round Time}}
\end{aligned}$$

**Model Quality Metrics:**

$$\begin{aligned}
\text{Global Accuracy} &= \text{Aggregated test accuracy} \\
\text{Client Variance} &= \text{Var}(\text{Client Accuracies}) \\
\text{Model Stability} &= 1 - \frac{\text{Var}(\text{Round Accuracies})}{\text{Mean}(\text{Round Accuracies})}
\end{aligned}$$

**System Metrics:**

$$\begin{aligned}
\text{Communication Overhead} &= \frac{\text{Total Bytes}}{\text{Model Parameters}} \\
\text{Training Time} &= \sum \text{Round Times} \\
\text{Client Dropout Rate} &= \frac{\text{Clients Dropped}}{\text{Total Clients}}
\end{aligned}$$

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

1. **Add MLflow:** Integrate experiment tracking
2. **Model versioning:** Implement round-based versioning
3. **Enhanced logging:** Structured logging for FL rounds

### Phase 2: Automation (Weeks 3-4)

1. **CI/CD pipeline:** Automated FL training
2. **Model registry:** Centralized model management
3. **Automated evaluation:** Post-training validation

### Phase 3: Production (Weeks 5-6)

1. **Model deployment:** API endpoints for inference
2. **Production monitoring:** Real-time performance tracking
3. **Automated retraining:** Trigger-based FL rounds

## Conclusion

MLOps is **highly applicable** to Federated Learning, but requires adaptations:

- **Experiment tracking** → Track FL rounds, not just final model
- **Model versioning** → Version by round, track convergence
- **Monitoring** → FL-specific metrics (convergence, communication)
- **Deployment** → Consider federated vs centralized deployment
- **Privacy** → Privacy-preserving monitoring and metrics

**Key Insight:** MLOps for FL focuses on the **aggregation process** and **round-based training**, not just the final model. The entire FL lifecycle needs operationalization.

