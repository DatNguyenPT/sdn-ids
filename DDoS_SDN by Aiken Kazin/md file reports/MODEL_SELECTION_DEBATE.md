# Architectural Selection Debate: MLP, LSTM, and CNN1D for DDoS Detection

**Analyzing Model Choices for Tabular Network Flow Data Classification**

## Problem Context

The task is **binary classification** of network flow data for DDoS attack detection:

- **Data Type:** Tabular/structured data (network flow features)
- **Features:** Packet count, byte count, duration, flows, packet rate, etc.
- **Task:** Classify each sample as Normal (0) or DDoS Attack (1)
- **Challenge:** Choosing appropriate neural network architecture for tabular data

## The Fundamental Question

**Why use MLP, LSTM, and CNN1D when the data is tabular, not sequential or image-based?**

This document presents arguments for and against each architecture choice.

## MLP (Multi-Layer Perceptron)

### Arguments FOR MLP

#### 1. Natural Fit for Tabular Data

- **Designed for structured data:** MLPs are the standard architecture for tabular/structured data
- **No assumptions:** Doesn't assume spatial or temporal structure in features
- **Feature interactions:** Fully connected layers can model any feature interaction
- **Interpretability:** Easier to understand and debug than CNNs/LSTMs

#### 2. Theoretical Justification

$$\hat{y} = \text{softmax}(W_2 \cdot \text{ReLU}(W_1 \cdot x + b_1) + b_2)$$

- Universal function approximator (can approximate any continuous function)
- No architectural constraints on feature relationships
- Direct mapping from features to output

#### 3. Computational Efficiency

- **Fast training:** Parallel processing, no sequential dependencies
- **Fewer parameters:** More parameter-efficient than CNN/LSTM
- **Lower memory:** Simpler architecture requires less memory
- **Scalability:** Scales well with data size

#### 4. Modern Enhancements

- **BatchNormalization:** Stabilizes training, allows higher learning rates
- **Dropout:** Prevents overfitting effectively
- **Regularization:** Well-regularized MLPs are competitive with complex models

### Arguments AGAINST MLP

#### 1. Limited Hierarchical Learning

- **No feature abstraction:** May not learn hierarchical feature representations
- **Feature engineering dependent:** Relies heavily on good feature engineering
- **Flat structure:** All features treated equally, no multi-scale patterns

#### 2. Curse of Dimensionality

- Can struggle with very high-dimensional feature spaces
- May not handle sparse features well
- Requires careful feature selection

### MLP Verdict

**✅ BEST CHOICE** for tabular data classification. Theoretically sound, computationally efficient, and appropriate for the problem domain.

## LSTM (Long Short-Term Memory)

### Arguments FOR LSTM

#### 1. Sequential Feature Processing

- **Sequential processing:** Processes features sequentially, potentially learning dependencies
- **Memory mechanism:** Can "remember" important features while processing others
- **Stateful learning:** Maintains hidden state that could encode feature relationships

#### 2. Temporal Interpretation

- **Network flow as temporal:** Network flows have temporal characteristics (packets arrive over time)
- **Feature ordering:** If features are ordered by importance or temporal relevance, LSTM can leverage this
- **Dependency modeling:** Can model complex dependencies between features through hidden states

#### 3. Empirical Success

- LSTMs have been used successfully in network intrusion detection
- Can model complex decision boundaries
- Proven in security domain applications

#### 4. Mathematical Formulation

$$\begin{aligned}
h_t &= \text{LSTM}(x_t, h_{t-1}, c_{t-1}) \\
\hat{y} &= \text{softmax}(W \cdot h_T + b)
\end{aligned}$$

- Processes features as sequence: $x_1, x_2, \ldots, x_T$
- Maintains memory through hidden state $h_t$ and cell state $c_t$
- Final hidden state $h_T$ captures sequential information

### Arguments AGAINST LSTM

#### 1. No True Temporal Structure

- **Tabular data is IID:** Each sample is independent—no temporal dependencies between samples
- **Feature vector $\neq$ time series:** Treating features as timesteps is conceptually incorrect
- **Misuse of architecture:** LSTM is designed for sequences where order matters (time series, text), not feature vectors

#### 2. Computational Overhead

- **More parameters:** LSTM has significantly more parameters than MLP

$$\text{LSTM params} = 4 \times (d \times d + d \times d_{in} + d)$$

where $d = \text{units}$, $d_{in} = \text{input size}$

- **Slower training:** Sequential processing is slower than parallel MLP processing
- **Vanishing gradients:** Can suffer from gradient issues in deep networks

#### 3. No Clear Benefit

- **Feature relationships:** MLPs can learn feature interactions through fully connected layers
- **Attention mechanisms:** Modern tabular models (e.g., TabNet) use attention, not RNNs
- **Over-engineering:** More complex than necessary for tabular data

#### 4. Feature Order Dependency

- **Order sensitivity:** Changing feature order changes LSTM behavior
- **Arbitrary ordering:** Features like `port_no`, `protocol`, `tx_kbps` don't have inherent ordering
- **Inconsistent results:** Different feature orders may yield different results

### LSTM Verdict

**⚠️ QUESTIONABLE CHOICE** for tabular data. Over-engineered for the problem, but may work empirically if features have meaningful ordering.

## CNN1D (1D Convolutional Neural Network)

### Arguments FOR CNN1D

#### 1. Feature Pattern Recognition

- **Local patterns:** CNN1D can detect local patterns and correlations between adjacent features
- **Example:** Features like `pktcount`, `bytecount`, `duration_sec` might have meaningful relationships when positioned together
- **Translation invariance:** CNN learns patterns regardless of exact feature positions (though less relevant for tabular data)

#### 2. Hierarchical Feature Learning

- **Multi-scale patterns:** Multiple Conv1D layers with pooling can learn hierarchical representations
- **Feature abstraction:** Lower layers detect simple patterns (e.g., "high packet count"), higher layers detect complex combinations
- **Reduced overfitting:** Convolutional layers share weights, reducing parameters compared to fully connected layers

#### 3. Mathematical Formulation

$$y[i] = \sum_{j=0}^{k-1} w[j] \cdot x[i+j] + b$$

- **Convolution operation:** Detects patterns in local windows
- **Multiple filters:** Each filter learns different patterns
- **Feature maps:** Output represents detected patterns at each position

#### 4. Empirical Success

- CNNs have shown success in tabular data tasks when features are treated as sequences
- Can capture non-linear interactions between features more efficiently than MLPs
- Proven in various classification tasks

#### 5. Architecture Benefits

- **MaxPooling:** Extracts dominant features, reduces dimensionality
- **Parameter sharing:** Same filter used across all positions
- **Multi-layer abstraction:** Simple patterns → complex patterns → classification

### Arguments AGAINST CNN1D

#### 1. No Spatial Structure

- **Tabular data lacks spatial/temporal ordering:** Features like `port_no`, `protocol`, `tx_kbps` don't have inherent spatial relationships
- **Arbitrary feature order:** Changing feature order shouldn't affect results, but CNNs are sensitive to input order
- **Misaligned assumptions:** CNNs assume local patterns matter, but in tabular data, any feature can interact with any other

#### 2. Over-engineering

- **Unnecessary complexity:** MLPs are designed for tabular data and may be more appropriate
- **Parameter efficiency:** CNN1D might not be more efficient than MLP for this problem size
- **Feature engineering:** Still requires good feature engineering

#### 3. Data Transformation Artifact

- The `expand_dims(axis=2)` operation artificially creates a "channel" dimension
- This is a workaround, not a natural representation
- Features are not truly sequential or spatial

#### 4. Limited Receptive Field

- **Small kernel size:** Kernel size 3 only sees 3 adjacent features
- **Long-range dependencies:** May miss relationships between distant features
- **Requires depth:** Needs multiple layers to capture global patterns

### CNN1D Verdict

**⭐ EXPERIMENTAL CHOICE** for tabular data. May capture local feature patterns, but lacks theoretical justification for non-spatial data.

## Comparative Analysis

### Architectural Comparison

| **Aspect** | **MLP** | **LSTM** | **CNN1D** |
|------------|---------|----------|-----------|
| **Data Assumption** | None (IID) | Sequential | Spatial/Local |
| **Feature Order** | Independent | Dependent | Dependent |
| **Parameters** | Low | High | Medium |
| **Training Speed** | Fast | Slow | Medium |
| **Interpretability** | High | Medium | Low |
| **Theoretical Fit** | ✅ Excellent | ❌ Poor | ⚠️ Questionable |
| **Empirical Fit** | ✅ Good | ⚠️ Variable | ⚠️ Variable |

### Parameter Count Comparison

For $d_{\text{features}} = 20$ and binary classification:

$$\begin{align*}
\text{MLP:} &\quad \sim 3,500 \text{ parameters} \\
\text{LSTM:} &\quad \sim 29,400 \text{ parameters} \\
\text{CNN1D:} &\quad \sim 49,600 \text{ parameters}
\end{align*}$$

**Conclusion:** MLP is most parameter-efficient.

### Computational Complexity

$$\begin{align*}
\text{MLP:} &\quad O(d_{\text{features}} \times d_{\text{hidden}}) \quad \text{(parallel)} \\
\text{LSTM:} &\quad O(T \times d^2) \quad \text{(sequential, } T = d_{\text{features}}) \\
\text{CNN1D:} &\quad O(T \times k \times F) \quad \text{(convolution, } k = \text{kernel}, F = \text{filters})
\end{align*}$$

**Conclusion:** MLP has lowest computational complexity.

## Why All Three Models Were Chosen

### Research Methodology

The selection of MLP, LSTM, and CNN1D represents an **experimental approach**:

1. **Baseline (MLP):** Provides theoretically sound baseline for comparison
2. **Experimental (LSTM/CNN1D):** Tests whether sequential/spatial assumptions help despite being tabular data
3. **Comparative Analysis:** Enables empirical comparison of different architectural paradigms

### Valid Research Questions

- **Q1:** Can sequential processing (LSTM) improve tabular classification?
- **Q2:** Can local pattern detection (CNN1D) capture feature relationships?
- **Q3:** Do complex architectures outperform simple MLP for this problem?
- **Q4:** Is the added complexity justified by performance gains?

### Expected Outcomes

- **MLP:** Expected to perform well (baseline)
- **LSTM:** May work if features have meaningful ordering
- **CNN1D:** May work if local feature patterns exist
- **Best Model:** Determined empirically through experiments

## Recommendations

### For This Problem

1. **Start with MLP** as primary model (theoretically sound)
2. **Use LSTM/CNN1D** for comparison and ablation studies
3. **Focus on feature engineering** rather than complex architectures
4. **Consider alternatives:** TabNet, XGBoost, or Gradient Boosting might be more appropriate

### For Future Work

- **Feature importance analysis:** Determine if feature ordering matters
- **Architecture ablation:** Systematically test which components help
- **Hybrid approaches:** Combine MLP with attention mechanisms
- **Ensemble methods:** Combine predictions from all three models

## Conclusion

### Summary

- **MLP:** ✅ Theoretically sound, appropriate baseline
- **LSTM:** ⚠️ Questionable for tabular data, but worth testing
- **CNN1D:** ⚠️ Experimental, may capture local patterns

### Final Verdict

The architectural choices reflect **valid research methodology**:

- Testing whether architectures designed for other domains can be adapted to tabular data
- Empirical results should determine which model performs best
- Theoretical justification is weak for LSTM/CNN1D, but empirical validation is key

**Key Insight:** While MLP is theoretically optimal for tabular data, LSTM and CNN1D serve as experimental baselines to test if sequential/spatial assumptions provide empirical benefits, even without strong theoretical foundation.

## Mathematical Formulations

### MLP Forward Pass

$$\begin{aligned}
h_1 &= \text{ReLU}(W_1 \cdot x + b_1) \\
h_2 &= \text{ReLU}(W_2 \cdot h_1 + b_2) \\
\hat{y} &= \text{softmax}(W_3 \cdot h_2 + b_3)
\end{aligned}$$

### LSTM Forward Pass

$$\begin{aligned}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \\
\tilde{c}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \\
h_t &= o_t \odot \tanh(c_t) \\
\hat{y} &= \text{softmax}(W \cdot h_T + b)
\end{aligned}$$

### CNN1D Forward Pass

$$\begin{aligned}
y_1[i] &= \sum_{j=0}^{k-1} w_1[j] \cdot x[i+j] + b_1 \quad \text{(Conv1D Layer 1)} \\
p_1[i] &= \max(y_1[2i], y_1[2i+1]) \quad \text{(MaxPooling)} \\
y_2[i] &= \sum_{j=0}^{k-1} w_2[j] \cdot p_1[i+j] + b_2 \quad \text{(Conv1D Layer 2)} \\
p_2[i] &= \max(y_2[2i], y_2[2i+1]) \quad \text{(MaxPooling)} \\
z &= \text{Flatten}(p_2) \\
h &= \text{ReLU}(W_1 \cdot z + b_1) \\
\hat{y} &= \text{softmax}(W_2 \cdot h + b_2)
\end{aligned}$$

## References and Further Reading

- MLP: Universal function approximators for tabular data
- LSTM: Originally designed for sequential data (time series, text)
- CNN1D: Designed for 1D signals (audio, time series, sequences)
- Tabular Data: Traditional ML (XGBoost, Random Forest) often outperforms deep learning

