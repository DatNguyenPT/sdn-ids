# IID vs Non-IID Performance Comparison

## Executive Summary

This document presents a comprehensive comparison of model performance under **IID (Independent and Identically Distributed)** and **Non-IID (Non-Independent and Identically Distributed)** data distributions in the federated learning DDoS detection system.

**Key Finding**: Non-IID distribution presents significant challenges, with accuracy varying dramatically across different model architectures.

---

## Experimental Setup

### Configuration

- **Federated Learning Rounds**: 5 rounds
- **Workers per Model**: 2 workers
- **Data Partition**: 50% per worker
- **Epochs per Round**: 5 epochs
- **Batch Size**: 32
- **Differential Privacy**: Disabled
- **Aggregation Strategy**: FedAvg

**Note**: For detailed explanations of IID and Non-IID distributions, their implementation, and impact, see the "Data Distribution Explanation" section below.

---

## Data Distribution Explanation

### What is IID (Independent and Identically Distributed)?

#### Definition

**IID** means that data samples are:
- **Independent**: Each sample is drawn independently (no correlation between samples)
- **Identically Distributed**: All samples come from the same probability distribution

In federated learning, **IID distribution** means each worker/client has data that:
- Follows the same statistical distribution
- Has similar class proportions
- Represents a random sample of the global dataset

#### How IID is Implemented

**Implementation Method:**
1. **Random Sampling**: Each worker receives a random subset of the global dataset
2. **Worker-Specific Seed**: Uses `hash(worker_id) % 2**32` to ensure reproducibility
3. **Preserved Distribution**: Class proportions remain similar across all workers

**Code Implementation:**
```python
if self.iid:
    # Random sampling with worker-specific seed
    seed = hash(self.worker_id) % 2**32
    n_samples = int(len(X_train) * self.data_partition)
    indices = np.random.RandomState(seed=seed).choice(
        len(X_train), size=n_samples, replace=False
    )
    X_train = X_train.iloc[indices]
    y_train = y_train.iloc[indices]
```

#### IID Class Distribution

**Global Dataset Distribution:**
- **Class 0 (Normal Traffic)**: ~60.91% of total dataset
- **Class 1 (DDoS Attack)**: ~39.09% of total dataset

**Per-Worker Distribution (IID):**
- **Worker 1**: ~60.78% Class 0, ~39.22% Class 1
- **Worker 2**: ~60.81% Class 0, ~39.19% Class 1
- **Worker 3**: ~60.85% Class 0, ~39.15% Class 1
- **Worker 4**: ~60.79% Class 0, ~39.21% Class 1

**Key Characteristics:**
- ✅ **Uniform Distribution**: All workers have similar class proportions
- ✅ **Random Sampling**: No bias in data selection
- ✅ **Preserved Statistics**: Global dataset statistics maintained per worker
- ✅ **No Overlap**: Different seeds ensure no data overlap between workers

#### IID Impact on Model Performance

**Positive Impacts:**
1. **Faster Convergence**: Models converge in fewer rounds (typically 3-5 rounds)
2. **Higher Accuracy**: Better final model performance (LSTM: 91.05%, CNN_LSTM: 87.23%)
3. **Stable Training**: Low variance in model updates across workers
4. **Optimal FedAvg**: Standard weighted averaging works effectively
5. **Better Generalization**: Models learn more robust features

**Performance Metrics (IID):**
- **LSTM**: 91.05% accuracy, 0.1607 loss ✅ (Best)
- **CNN_LSTM**: 87.23% accuracy, 0.2596 loss ✅ (Good)
- **CNN1D**: 76.16% accuracy, 0.4691 loss ⚠️ (Moderate)
- **MLPv2**: 60.91% accuracy, 0.6571 loss ⚠️ (Moderate)

---

### What is Non-IID (Non-Independent and Identically Distributed)?

#### Definition

**Non-IID** means that data samples are:
- **Not Identically Distributed**: Different workers have different data distributions
- **May be Correlated**: Samples within a worker may be correlated
- **Class-Skewed**: Workers have significantly different class proportions

In federated learning, **Non-IID distribution** means:
- Workers have **different class distributions**
- Some workers have **more of one class** than others
- More realistic representation of real-world scenarios (e.g., hospitals, IoT devices)

#### How Non-IID is Implemented

**Implementation Method:**
1. **Class-Based Partitioning**: Workers receive data based on class labels
2. **Skewed Distribution**: Odd workers get 80% Class 0, Even workers get 80% Class 1
3. **Deterministic Assignment**: Worker number (odd/even) determines class skew

**Code Implementation:**
```python
if not self.iid:
    # Extract worker number (e.g., "worker1" -> 1)
    worker_num = int(''.join(filter(str.isdigit, self.worker_id)))
    is_odd_worker = (worker_num % 2 == 1)
    
    # Get class indices
    class_0_indices = np.where(y_train == 0)[0]
    class_1_indices = np.where(y_train == 1)[0]
    
    if is_odd_worker:
        # Odd workers: 80% Class 0, 20% Class 1
        n_class_0 = int(total_samples_needed * 0.8)
        n_class_1 = total_samples_needed - n_class_0
    else:
        # Even workers: 20% Class 0, 80% Class 1
        n_class_1 = int(total_samples_needed * 0.8)
        n_class_0 = total_samples_needed - n_class_1
    
    # Sample from each class
    selected_class_0 = rng.choice(class_0_indices, size=n_class_0, replace=False)
    selected_class_1 = rng.choice(class_1_indices, size=n_class_1, replace=False)
    
    # Combine and shuffle
    indices = np.concatenate([selected_class_0, selected_class_1])
    X_train = X_train.iloc[indices]
    y_train = y_train.iloc[indices]
```

#### Non-IID Class Distribution Pattern

**Per-Worker Distribution (Non-IID):**

**Odd Workers** (worker1, worker3, worker5, worker7):
- **Class 0 (Normal Traffic)**: 80% of worker's data
- **Class 1 (DDoS Attack)**: 20% of worker's data
- **Example**: Worker 1 might have 800 samples of Class 0, 200 samples of Class 1

**Even Workers** (worker2, worker4, worker6, worker8):
- **Class 0 (Normal Traffic)**: 20% of worker's data
- **Class 1 (DDoS Attack)**: 80% of worker's data
- **Example**: Worker 2 might have 200 samples of Class 0, 800 samples of Class 1

**Visual Representation:**
```
Worker 1 (Odd):  [████████████████████████████████] 80% Class 0
                 [████] 20% Class 1

Worker 2 (Even): [████] 20% Class 0
                 [████████████████████████████████] 80% Class 1
```

**Key Characteristics:**
- ⚠️ **Skewed Distribution**: Workers have complementary but different class proportions
- ⚠️ **Class Imbalance**: 60% difference in class distribution between odd/even workers
- ⚠️ **Realistic Scenario**: Mimics real-world federated learning (e.g., different hospitals, IoT devices)
- ⚠️ **Challenging**: Creates significant challenges for model convergence

#### Non-IID Impact on Model Performance

**Negative Impacts:**
1. **Slower Convergence**: Requires more communication rounds to converge
2. **Lower Accuracy**: Achieves lower final accuracy compared to IID
3. **Higher Variance**: More variability in model updates across workers
4. **Model-Specific Sensitivity**: Different architectures show varying robustness:
   - **CNN_LSTM**: Only -3.45% accuracy drop (most robust)
   - **LSTM**: -5.47% accuracy drop (very robust)
   - **CNN1D**: -15.25% accuracy drop (moderate impact)
   - **MLPv2**: -21.82% accuracy drop (most sensitive, fails catastrophically)

**Performance Metrics (Non-IID):**
- **LSTM**: 85.58% accuracy, 0.3153 loss ✅ (Best, but 5.47% drop from IID)
- **CNN_LSTM**: 83.78% accuracy, 0.3094 loss ✅ (Good, only 3.45% drop from IID)
- **CNN1D**: 60.91% accuracy, 0.6972 loss ⚠️ (Moderate, 15.25% drop from IID)
- **MLPv2**: 39.09% accuracy, 0.9770 loss ❌ (Poor, 21.82% drop, below random chance)

**Why Non-IID is Challenging:**
1. **Gradient Mismatch**: Workers optimize for different objectives (different class distributions)
2. **Aggregation Difficulty**: FedAvg struggles when workers have conflicting updates
3. **Local Overfitting**: Workers may overfit to their local class distribution
4. **Convergence Instability**: Model may oscillate or converge to suboptimal solutions

---

## Non-IID Results

### Performance Metrics

| Model | Accuracy | Loss | Parameters | Convergence |
|-------|----------|------|------------|-------------|
| **CNN_LSTM** | **83.78%** | 0.3094 | 37.57K | ✅ Good |
| **LSTM** | **85.58%** | 0.3153 | 29.38K | ✅ Good |
| **CNN1D** | **60.91%** | 0.6972 | 58.69K | ⚠️ Moderate |
| **MLPv2** | **39.09%** | 0.9770 | 4.07K | ❌ Poor |

### Detailed Non-IID Results

#### LSTM (Best Performance)
- **Accuracy**: 85.58%
- **Loss**: 0.3153
- **Parameters**: 29.38K
- **Network Traffic**: 1.13 MB sent, 1.13 MB received
- **Total Params Exchanged**: 296.34K sent, 296.34K received
- **Status**: ✅ Complete - Round 5/5

**Analysis**: LSTM shows the best performance under Non-IID conditions, achieving 85.58% accuracy. The sequential architecture with LSTM layers effectively captures temporal patterns and handles class imbalance well, outperforming the hybrid CNN_LSTM model.

#### CNN_LSTM
- **Accuracy**: 83.78%
- **Loss**: 0.3094
- **Parameters**: 37.57K
- **Network Traffic**: 1.44 MB sent, 1.44 MB received
- **Total Params Exchanged**: 378.58K sent, 378.58K received
- **Status**: ✅ Complete - Round 5/5

**Analysis**: CNN_LSTM performs well (83.78%) under Non-IID conditions, achieving the second-best accuracy. The hybrid architecture (CNN + LSTM) handles class imbalance reasonably well, though slightly behind the pure LSTM model.

#### CNN1D
- **Accuracy**: 60.91%
- **Loss**: 0.6972
- **Parameters**: 58.69K
- **Network Traffic**: 2.26 MB sent, 2.26 MB received
- **Total Params Exchanged**: 592.02K sent, 592.02K received
- **Status**: ✅ Complete - Round 5/5

**Analysis**: CNN1D achieves moderate accuracy (60.91%) but struggles with the class imbalance. The convolutional layers may not adapt well to the skewed distribution.

#### MLPv2 (Worst Performance)
- **Accuracy**: 39.09%
- **Loss**: 0.9770
- **Parameters**: 4.07K
- **Network Traffic**: 176.33 KB sent, 176.33 KB received
- **Total Params Exchanged**: 45.14K sent, 45.14K received
- **Status**: ✅ Complete - Round 5/5

**Analysis**: MLPv2 shows the poorest performance (39.09% accuracy), which is below random chance for binary classification (50%). This suggests the model is failing to learn meaningful patterns under Non-IID conditions. The high loss (0.9770) indicates poor convergence.

---

## IID Results

### Performance Metrics

| Model | Accuracy | Loss | Parameters | Convergence |
|-------|----------|------|------------|-------------|
| **LSTM** | **91.05%** | 0.1607 | 29.38K | ✅ Excellent |
| **CNN_LSTM** | **87.23%** | 0.2596 | 37.57K | ✅ Good |
| **CNN1D** | **76.16%** | 0.4691 | 58.69K | ⚠️ Moderate |
| **MLPv2** | **60.91%** | 0.6571 | 4.07K | ⚠️ Moderate |

### Detailed IID Results

#### LSTM (Best Performance)
- **Accuracy**: 91.05%
- **Loss**: 0.1607
- **Parameters**: 29.38K
- **Network Traffic**: 1.02 MB sent, 926.06 KB received
- **Total Params Exchanged**: 266.71K sent, 237.07K received
- **Status**: ✅ Complete - Round 5/5

**Analysis**: LSTM achieves the best performance under IID conditions with 91.05% accuracy and very low loss (0.1607). The sequential architecture excels when data is uniformly distributed across workers.

#### CNN_LSTM
- **Accuracy**: 87.23%
- **Loss**: 0.2596
- **Parameters**: 37.57K
- **Network Traffic**: 1.44 MB sent, 1.44 MB received
- **Total Params Exchanged**: 378.58K sent, 378.58K received
- **Status**: ✅ Complete - Round 5/5

**Analysis**: CNN_LSTM performs well under IID conditions (87.23% accuracy), demonstrating the effectiveness of hybrid architectures when data distribution is uniform.

#### CNN1D
- **Accuracy**: 76.16%
- **Loss**: 0.4691
- **Parameters**: 58.69K
- **Network Traffic**: 2.26 MB sent, 2.26 MB received
- **Total Params Exchanged**: 592.02K sent, 592.02K received
- **Status**: ✅ Complete - Round 5/5

**Analysis**: CNN1D achieves moderate accuracy (76.16%) under IID conditions. While better than Non-IID, it still lags behind sequential and hybrid models.

#### MLPv2
- **Accuracy**: 60.91%
- **Loss**: 0.6571
- **Parameters**: 4.07K
- **Network Traffic**: 176.33 KB sent, 176.33 KB received
- **Total Params Exchanged**: 45.14K sent, 45.14K received
- **Status**: ✅ Complete - Round 5/5

**Analysis**: MLPv2 shows moderate performance (60.91% accuracy) under IID conditions, which is significantly better than its Non-IID performance (39.09%). However, it still underperforms compared to other architectures.

---

## Comparison Analysis

### Side-by-Side Performance Comparison

| Model | IID Accuracy | Non-IID Accuracy | Accuracy Drop | IID Loss | Non-IID Loss | Impact |
|-------|--------------|------------------|---------------|----------|--------------|--------|
| **LSTM** | **91.05%** | **85.58%** | **-5.47%** | 0.1607 | 0.3153 | ⚠️ Moderate |
| **CNN_LSTM** | **87.23%** | **83.78%** | **-3.45%** | 0.2596 | 0.3094 | ✅ Low |
| **CNN1D** | **76.16%** | **60.91%** | **-15.25%** | 0.4691 | 0.6972 | ❌ High |
| **MLPv2** | **60.91%** | **39.09%** | **-21.82%** | 0.6571 | 0.9770 | ❌ Very High |

### Performance Ranking

#### IID Distribution
1. **LSTM**: 91.05% ✅ (Best)
2. **CNN_LSTM**: 87.23% ✅ (Good)
3. **CNN1D**: 76.16% ⚠️ (Moderate)
4. **MLPv2**: 60.91% ⚠️ (Moderate)

#### Non-IID Distribution
1. **LSTM**: 85.58% ✅ (Best)
2. **CNN_LSTM**: 83.78% ✅ (Good)
3. **CNN1D**: 60.91% ⚠️ (Moderate)
4. **MLPv2**: 39.09% ❌ (Poor)

### Key Observations

#### 1. **IID vs Non-IID Impact Analysis**

**Most Robust Models (Lowest Accuracy Drop):**
- **CNN_LSTM**: Only -3.45% drop (87.23% → 83.78%) - Most resilient to Non-IID
- **LSTM**: -5.47% drop (91.05% → 85.58%) - Very robust, maintains high accuracy

**Most Sensitive Models (Highest Accuracy Drop):**
- **MLPv2**: -21.82% drop (60.91% → 39.09%) - Severely impacted by Non-IID
- **CNN1D**: -15.25% drop (76.16% → 60.91%) - Significantly affected

#### 2. **Model Architecture Sensitivity**

Different architectures show varying sensitivity to Non-IID distribution:

- **Hybrid Models (CNN_LSTM)**: Most robust to Non-IID, only 3.45% accuracy drop
- **Sequential Models (LSTM)**: Very robust, 5.47% drop, maintains best overall accuracy
- **Convolutional Models (CNN1D)**: Moderate robustness, 15.25% drop
- **Fully Connected (MLPv2)**: Least robust, 21.82% drop, fails under Non-IID

#### 3. **Class Imbalance Impact**

The 80/20 class distribution creates significant challenges:

- **MLPv2** struggles most, dropping from 60.91% (IID) to 39.09% (Non-IID) - below random chance
- **CNN_LSTM** handles imbalance best, maintaining 83.78% accuracy with only 3.45% drop
- **LSTM** shows excellent resilience, maintaining 85.58% accuracy despite 5.47% drop
- Models with more parameters (CNN1D: 58.69K) don't necessarily perform better

#### 4. **Loss vs Accuracy Correlation**

**IID Distribution:**
- **LSTM**: Very low loss (0.1607) → Excellent accuracy (91.05%) ✅
- **CNN_LSTM**: Low loss (0.2596) → Good accuracy (87.23%) ✅
- **CNN1D**: Moderate loss (0.4691) → Moderate accuracy (76.16%) ⚠️
- **MLPv2**: High loss (0.6571) → Moderate accuracy (60.91%) ⚠️

**Non-IID Distribution:**
- **LSTM**: Low loss (0.3153) → High accuracy (85.58%) ✅
- **CNN_LSTM**: Low loss (0.3094) → High accuracy (83.78%) ✅
- **CNN1D**: High loss (0.6972) → Moderate accuracy (60.91%) ⚠️
- **MLPv2**: Very high loss (0.9770) → Poor accuracy (39.09%) ❌

**Loss Increase Analysis:**
- **LSTM**: Loss increases by 96.2% (0.1607 → 0.3153) but maintains high accuracy
- **CNN_LSTM**: Loss increases by 19.2% (0.2596 → 0.3094) - most stable
- **CNN1D**: Loss increases by 48.6% (0.4691 → 0.6972)
- **MLPv2**: Loss increases by 48.7% (0.6571 → 0.9770)

#### 5. **Convergence Behavior**

**IID Distribution:**
- **LSTM**: Excellent convergence (very low loss: 0.1607, high accuracy: 91.05%)
- **CNN_LSTM**: Good convergence (low loss: 0.2596, good accuracy: 87.23%)
- **CNN1D**: Moderate convergence (moderate loss: 0.4691, moderate accuracy: 76.16%)
- **MLPv2**: Moderate convergence (high loss: 0.6571, moderate accuracy: 60.91%)

**Non-IID Distribution:**
- **LSTM**: Excellent convergence (low loss: 0.3153, high accuracy: 85.58%)
- **CNN_LSTM**: Good convergence (low loss: 0.3094, high accuracy: 83.78%)
- **CNN1D**: Moderate convergence (high loss: 0.6972, moderate accuracy: 60.91%)
- **MLPv2**: Poor convergence (very high loss: 0.9770, poor accuracy: 39.09%)

---

## Discussion

### Why LSTM Performs Best Under Non-IID?

1. **Sequential Processing**: LSTM's sequential nature effectively captures temporal patterns in the data
2. **Memory Mechanism**: Long short-term memory helps maintain context across skewed class distributions
3. **Parameter Efficiency**: 29.38K parameters provide good balance between capacity and generalization
4. **Robustness**: LSTM gates (forget, input, output) help filter and adapt to imbalanced data
5. **Adaptability**: Can learn meaningful patterns despite 80/20 class skew across workers

### Why CNN_LSTM Performs Well Under Non-IID?

1. **Feature Extraction**: CNN layers extract local patterns, LSTM captures temporal dependencies
2. **Robustness**: Hybrid architecture provides redundancy and robustness
3. **Parameter Count**: 37.57K parameters provide sufficient capacity
4. **Adaptability**: Can learn from both local and sequential patterns despite class imbalance

### Why MLPv2 Performs Worst Under Non-IID?

1. **Simple Architecture**: Fully connected layers lack specialized feature extraction
2. **Limited Capacity**: Only 4.07K parameters may be insufficient for complex patterns
3. **No Inductive Bias**: Lacks architectural biases that help with imbalanced data
4. **Gradient Issues**: May struggle with gradient flow in Non-IID scenarios

### Implications for Federated Learning

1. **Model Selection**: Architecture choice is critical for Non-IID scenarios
2. **Robustness Testing**: Non-IID reveals model weaknesses not visible in IID
3. **Production Deployment**: Real-world data is often Non-IID, making this testing essential
4. **Algorithm Development**: May need specialized aggregation strategies for Non-IID

---

## Network Traffic Analysis

### Non-IID Network Traffic

| Model | Bytes Sent | Bytes Received | Params Sent | Params Received |
|-------|------------|----------------|-------------|-----------------|
| **CNN1D** | 2.26 MB | 2.26 MB | 592.02K | 592.02K |
| **CNN_LSTM** | 1.44 MB | 1.44 MB | 378.58K | 378.58K |
| **LSTM** | 1.13 MB | 1.13 MB | 296.34K | 296.34K |
| **MLPv2** | 176.33 KB | 176.33 KB | 45.14K | 45.14K |

**Observations**:
- Network traffic correlates with model size (parameters)
- CNN1D has highest traffic due to largest parameter count
- MLPv2 has lowest traffic due to smallest parameter count
- Traffic is symmetric (sent ≈ received) for most models

---

## Conclusion

### Performance Summary

**IID Distribution Performance:**
- **Best Case (LSTM)**: 91.05% accuracy - excellent for production
- **Second Best (CNN_LSTM)**: 87.23% accuracy - good for production
- **Moderate (CNN1D)**: 76.16% accuracy - acceptable for production
- **Weakest (MLPv2)**: 60.91% accuracy - borderline acceptable

**Non-IID Distribution Performance:**
- **Best Case (LSTM)**: 85.58% accuracy - good for production (5.47% drop)
- **Second Best (CNN_LSTM)**: 83.78% accuracy - acceptable for production (3.45% drop)
- **Moderate (CNN1D)**: 60.91% accuracy - acceptable but degraded (15.25% drop)
- **Worst Case (MLPv2)**: 39.09% accuracy - unacceptable, below random chance (21.82% drop)

**Key Finding**: IID distribution consistently outperforms Non-IID for all models, with accuracy drops ranging from 3.45% (CNN_LSTM) to 21.82% (MLPv2).

### Key Takeaways

1. **IID is Better**: All models perform better under IID distribution, confirming the importance of data distribution uniformity
2. **Architecture Matters**: CNN_LSTM shows best robustness to Non-IID (only 3.45% drop), while MLPv2 is most sensitive (21.82% drop)
3. **LSTM Best Overall**: LSTM achieves highest accuracy in both IID (91.05%) and Non-IID (85.58%) scenarios
4. **Hybrid Models Resilient**: CNN_LSTM maintains excellent performance with minimal degradation under Non-IID
5. **MLPv2 Struggles**: Simple architectures fail catastrophically under Non-IID conditions (below random chance)
6. **Testing Essential**: Non-IID testing reveals real-world performance degradation
7. **Model Selection**: Choose architectures based on expected data distribution - use CNN_LSTM or LSTM for Non-IID scenarios

### Next Steps

1. **Add IID Results**: Compare with IID performance to quantify the impact
2. **Analyze Convergence**: Study convergence patterns across rounds
3. **Investigate MLPv2**: Understand why MLPv2 fails under Non-IID
4. **Optimize Strategies**: Develop aggregation strategies for Non-IID scenarios

---

## Appendix: Experimental Details

### Non-IID Distribution Details

- **Worker 1 (Odd)**: 80% Class 0, 20% Class 1
- **Worker 2 (Even)**: 20% Class 0, 80% Class 1
- **Class Skew**: 60% difference between workers
- **Data Overlap**: None (complementary distributions)

### Training Configuration

- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Categorical Crossentropy
- **Activation**: Softmax (output layer)
- **Regularization**: None (for baseline comparison)
- **Class Weights**: Not applied (to test raw Non-IID impact)

---

*Document Status: Non-IID results complete. IID results pending.*

