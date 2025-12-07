# Differential Privacy Parameters in Federated Learning DDoS Detection

## Introduction

This document explains the Differential Privacy (DP) parameters used in our Federated Learning system for DDoS detection. We implement DP-SGD (Differentially Private Stochastic Gradient Descent) to protect individual training samples while maintaining model utility.

## Differential Privacy Parameters

### Overview

Our implementation uses four key parameters to control the privacy-accuracy trade-off:

- **Clip Norm** ($C$): Gradient clipping threshold
- **Noise Multiplier** ($z$): Controls noise magnitude
- **Delta** ($\delta$): Failure probability
- **Epsilon** ($\epsilon$): Privacy budget (calculated)

---

## 1. Clip Norm (dp_clip_norm = 1.0)

### Definition

Clip Norm, denoted as $C$, is the maximum L2 norm allowed for gradient vectors. It bounds the sensitivity of gradients to individual training samples.

### Mathematical Formulation

For a gradient vector $\mathbf{g}$, clipping is performed as:

$$\mathbf{g}_{\text{clipped}} = \mathbf{g} \cdot \min\left(1, \frac{C}{\|\mathbf{g}\|_2}\right)$$

where $\|\mathbf{g}\|_2$ is the L2 norm of the gradient.

### Implementation in Our Project

In `flower_worker.py`, clip norm is set during worker initialization:

```python
dp_clip_norm: float = 1.0  # Default value
```

The DP-SGD optimizer uses this value:

```python
optimizer = DPKerasAdamOptimizer(
    l2_norm_clip=self.dp_clip_norm,  # C = 1.0
    noise_multiplier=self.dp_noise_multiplier,
    num_microbatches=1,
    learning_rate=0.001
)
```

### Impact

- **Lower $C$ (e.g., 0.5)**: Stronger privacy, slower convergence, lower accuracy
- **Higher $C$ (e.g., 2.0)**: Weaker privacy, faster convergence, higher accuracy
- **Our setting ($C = 1.0$)**: Balanced trade-off

### Example

```python
# Before clipping: gradient = [0.5, 0.8, 1.2]  (norm = 1.56)
# After clipping:  gradient = [0.32, 0.51, 0.77] (norm = 1.0)
```

---

## 2. Noise Multiplier (dp_noise_multiplier = 1.0)

### Definition

Noise Multiplier, denoted as $z$, controls the magnitude of Gaussian noise added to gradients. It determines the standard deviation of the noise distribution.

### Mathematical Formulation

After clipping, Gaussian noise is added:

$$\mathbf{g}_{\text{noisy}} = \mathbf{g}_{\text{clipped}} + \mathcal{N}(0, (C \cdot z)^2 \mathbf{I})$$

where:
- $C$ is the clip norm
- $z$ is the noise multiplier
- $\mathcal{N}(0, \sigma^2)$ is a Gaussian distribution with mean 0 and variance $\sigma^2$

With our settings ($C = 1.0$, $z = 1.0$), noise variance is:

$$\sigma^2 = (1.0 \times 1.0)^2 = 1.0$$

### Implementation

```python
dp_noise_multiplier: float = 1.0  # Default value

optimizer = DPKerasAdamOptimizer(
    l2_norm_clip=self.dp_clip_norm,
    noise_multiplier=self.dp_noise_multiplier,  # z = 1.0
    ...
)
```

### Impact

- **Lower $z$ (e.g., 0.5)**: Less noise, weaker privacy, higher accuracy
- **Higher $z$ (e.g., 2.0)**: More noise, stronger privacy, lower accuracy
- **Our setting ($z = 1.0$)**: Standard noise level

### Typical Values

- **0.1-0.5**: Weak privacy, better accuracy
- **1.0-2.0**: Moderate privacy (our setting)
- **2.0+**: Strong privacy, lower accuracy

---

## 3. Delta (dp_delta = 1e-5)

### Definition

Delta ($\delta$) is the failure probability in the $(\epsilon, \delta)$-differential privacy guarantee. It represents the probability that the privacy guarantee fails.

### Mathematical Meaning

The $(\epsilon, \delta)$-DP guarantee states:

$$\mathbb{P}[M(D) \in S] \leq e^{\epsilon} \cdot \mathbb{P}[M(D') \in S] + \delta$$

where:
- $M$ is the mechanism (our training algorithm)
- $D$ and $D'$ are datasets differing by one sample
- $S$ is any subset of possible outputs
- $\delta$ is the failure probability

### Implementation

```python
dp_delta: float = 1e-5  # Default: 0.00001 = 0.001%

epsilon_this_round, _ = compute_dp_sgd_privacy(
    n=num_samples,
    batch_size=self.batch_size,
    noise_multiplier=self.dp_noise_multiplier,
    epochs=self.epochs_per_round,
    delta=self.dp_delta  # δ = 1e-5
)
```

### Interpretation

With $\delta = 10^{-5}$:
- Privacy guarantee holds with probability $1 - \delta = 99.999\%$
- Failure probability: 0.001%
- Standard choice for datasets with $\sim 10^5$ samples

### Common Values

- **$\delta = 1/n$**: Where $n$ is the number of training samples
- **$\delta = 10^{-5}$**: Standard for medium datasets (our choice)
- **$\delta = 10^{-6}$**: Stricter, for larger datasets

---

## 4. Epsilon (ε) - Privacy Budget

### Definition

Epsilon ($\epsilon$) is the privacy loss parameter. It quantifies how much information about individual samples can be inferred from the model output.

### Calculation

Epsilon is calculated using the DP-SGD privacy accounting formula:

$$\epsilon = f(n, \text{batch\_size}, z, \text{epochs}, \delta)$$

where:
- $n$: Number of training samples
- $\text{batch\_size}$: Samples per batch (32 in our case)
- $z$: Noise multiplier (1.0)
- $\text{epochs}$: Training epochs per round (5)
- $\delta$: Failure probability (1e-5)

### Implementation

In our code, epsilon is computed after each training round:

```python
epsilon_this_round, _ = compute_dp_sgd_privacy(
    n=num_samples,              # e.g., 41,738 samples
    batch_size=self.batch_size,  # 32
    noise_multiplier=self.dp_noise_multiplier,  # 1.0
    epochs=self.epochs_per_round,  # 5
    delta=self.dp_delta  # 1e-5
)
self.epsilon_spent += epsilon_this_round
```

### Current Values

With our settings:
- Samples per worker: $\sim 41,738$
- Batch size: 32
- Epochs per round: 5
- Noise multiplier: 1.0
- **Epsilon per round**: $\approx 0.7$
- **Total epsilon (5 rounds)**: $\approx 3.5$

### Interpretation

- **$\epsilon = 0$**: Perfect privacy (no information released)
- **$\epsilon = 1$**: Very strong privacy
- **$\epsilon = 3-5$**: Moderate privacy (our range)
- **$\epsilon = 10$**: Weak privacy
- **$\epsilon > 100$**: Essentially no privacy

---

## How Parameters Work Together

### DP-SGD Algorithm Flow

The complete DP-SGD process in our implementation:

1. **Forward Pass**: Compute predictions and loss
2. **Backward Pass**: Compute gradients $\mathbf{g}$
3. **Clipping**: $\mathbf{g}_{\text{clipped}} = \text{clip}(\mathbf{g}, C)$
4. **Noise Addition**: $\mathbf{g}_{\text{noisy}} = \mathbf{g}_{\text{clipped}} + \mathcal{N}(0, (C \cdot z)^2)$
5. **Update**: $\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \cdot \mathbf{g}_{\text{noisy}}$
6. **Privacy Accounting**: Compute $\epsilon$ spent

### Privacy-Accuracy Trade-off

| Parameter Change | Privacy | Accuracy | Convergence |
|-----------------|---------|----------|-------------|
| Increase Clip Norm ($C \uparrow$) | $\downarrow$ | $\uparrow$ | $\uparrow$ |
| Decrease Clip Norm ($C \downarrow$) | $\uparrow$ | $\downarrow$ | $\downarrow$ |
| Increase Noise Mult ($z \uparrow$) | $\uparrow$ | $\downarrow$ | $\downarrow$ |
| Decrease Noise Mult ($z \downarrow$) | $\downarrow$ | $\uparrow$ | $\uparrow$ |

---

## Implementation in Our Project

### Worker-Side DP

DP is implemented at the worker level in `flower_worker.py`:

```python
class FlowerWorker(fl.client.NumPyClient):
    def __init__(self, ..., enable_dp=False, 
                 dp_clip_norm=1.0, 
                 dp_noise_multiplier=1.0, 
                 dp_delta=1e-5):
        self.enable_dp = enable_dp and DP_AVAILABLE
        self.dp_clip_norm = dp_clip_norm
        self.dp_noise_multiplier = dp_noise_multiplier
        self.dp_delta = dp_delta
        self.epsilon_spent = 0.0
```

### DP Optimizer Usage

When DP is enabled, the model uses DP-SGD optimizer:

```python
if self.enable_dp:
    optimizer = DPKerasAdamOptimizer(
        l2_norm_clip=self.dp_clip_norm,      # C = 1.0
        noise_multiplier=self.dp_noise_multiplier,  # z = 1.0
        num_microbatches=1,
        learning_rate=0.001
    )
else:
    optimizer = Adam(learning_rate=0.001)
```

### Privacy Budget Tracking

After each training round, epsilon is computed and accumulated:

```python
if self.enable_dp and compute_dp_sgd_privacy is not None:
    epsilon_this_round, _ = compute_dp_sgd_privacy(
        n=num_samples,
        batch_size=self.batch_size,
        noise_multiplier=self.dp_noise_multiplier,
        epochs=self.epochs_per_round,
        delta=self.dp_delta
    )
    self.epsilon_spent += epsilon_this_round
```

### Environment Variable Control

DP can be enabled/disabled via environment variable:

```bash
# Enable DP
FL_ENABLE_DP=true docker compose up -d

# Disable DP (default)
FL_ENABLE_DP=false docker compose up -d
```

---

## Privacy Budget Accumulation

### Per-Round Epsilon

Each training round consumes privacy budget:

$$\epsilon_{\text{round}} = f(n, B, z, E, \delta)$$

where:
- $n$: Samples per worker (41,738)
- $B$: Batch size (32)
- $z$: Noise multiplier (1.0)
- $E$: Epochs per round (5)
- $\delta$: Delta (1e-5)

### Total Privacy Budget

After $R$ rounds:

$$\epsilon_{\text{total}} = \sum_{r=1}^{R} \epsilon_{\text{round}} = R \times \epsilon_{\text{round}}$$

For our 5-round training:
$$\epsilon_{\text{total}} \approx 5 \times 0.7 = 3.5$$

---

## Example: Privacy Budget Calculation

### Scenario

- Worker 1: 41,738 samples
- Batch size: 32
- Epochs per round: 5
- Noise multiplier: 1.0
- Delta: 1e-5
- Total rounds: 5

### Calculation

Using the DP-SGD privacy accounting:

$$\epsilon_{\text{round}} \approx 0.6979$$
$$\epsilon_{\text{total}} = 5 \times 0.6979 = 3.4895$$

### Interpretation

After 5 rounds of federated training:
- Privacy guarantee: $(3.49, 10^{-5})$-DP
- Meaning: Adding/removing one sample changes output by factor $e^{3.49} \approx 32.8$
- Failure probability: 0.001%

---

## Recommendations

### For Stronger Privacy

- Reduce clip norm: $C = 0.5$
- Increase noise multiplier: $z = 2.0$
- Reduce epochs per round: $E = 3$
- Result: $\epsilon \approx 0.3$ per round, total $\approx 1.5$

### For Better Accuracy

- Increase clip norm: $C = 2.0$
- Reduce noise multiplier: $z = 0.5$
- Keep epochs: $E = 5$
- Result: $\epsilon \approx 1.2$ per round, total $\approx 6.0$

### For Balanced Trade-off

- Current settings: $C = 1.0$, $z = 1.0$, $E = 5$
- Result: $\epsilon \approx 0.7$ per round, total $\approx 3.5$
- Good balance between privacy and accuracy

---

## Summary

Our DP implementation uses four key parameters to control the privacy-accuracy trade-off:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| **Clip Norm** ($C$) | 1.0 | Maximum gradient magnitude |
| **Noise Multiplier** ($z$) | 1.0 | Amount of noise added |
| **Delta** ($\delta$) | 1e-5 | Failure probability (0.001%) |
| **Epsilon** ($\epsilon$) | ~0.7/round | Privacy loss per round |

These parameters work together to provide $(3.5, 10^{-5})$-differential privacy after 5 rounds of federated training, protecting individual training samples while maintaining reasonable model accuracy.

---

## Code References

- **Worker Implementation**: `flower_worker.py`
  - DP parameters initialization: Lines 73-90
  - DP optimizer setup: Lines 207-217
  - Privacy budget calculation: Lines 349-362

- **Environment Variable**: `docker-compose.yml`
  - `FL_ENABLE_DP=${FL_ENABLE_DP:-false}` in all worker services

- **Dashboard Display**: `fl_dashboard.py`
  - DP metrics visualization: Lines 283-372
  - Privacy budget chart: Lines 406-440

