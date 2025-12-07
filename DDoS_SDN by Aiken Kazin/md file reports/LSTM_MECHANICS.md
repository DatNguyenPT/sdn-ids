# LSTM Internal Mechanisms: Hidden State, Cell State, and Gates

**Detailed Explanation for DDoS Detection Model**

## Overview

The LSTM model in the code uses two LSTM layers:

```python
LSTM(64, return_sequences=True, input_shape=(num_features, 1))
LSTM(32, return_sequences=False)
Dense(num_classes, activation='softmax')
```

## Core Components

### Hidden State ($h_t$)

**Definition:** The hidden state is a vector that carries information from previous timesteps and represents the "short-term memory" of the LSTM.

**Mathematical Representation:**

$$h_t \in \mathbb{R}^d$$

where $d$ is the number of LSTM units (64 for Layer 1, 32 for Layer 2).

**In Your Code:**

- **Layer 1:** $h_t^{(1)} \in \mathbb{R}^{64}$ for each timestep $t$
- **Layer 2:** $h_t^{(2)} \in \mathbb{R}^{32}$ for each timestep $t$

**Flow Through Timesteps:**

$$h_0 \xrightarrow{x_1} h_1 \xrightarrow{x_2} h_2 \xrightarrow{x_3} h_3 \xrightarrow{\cdots} h_T$$

where $T = \text{num\_features}$ (typically 20 in your dataset).

**Output Behavior:**

- **Layer 1:** $\text{return\_sequences=True}$ $\Rightarrow$ outputs all $h_1^{(1)}, h_2^{(1)}, \ldots, h_T^{(1)}$
- **Layer 2:** $\text{return\_sequences=False}$ $\Rightarrow$ outputs only final $h_T^{(2)}$

### Cell State ($c_t$)

**Definition:** The cell state is the "long-term memory" that flows through the LSTM, storing information across many timesteps.

**Mathematical Representation:**

$$c_t \in \mathbb{R}^d$$

**Initialization:**

$$c_0 = \mathbf{0} \quad \text{(zero vector)}$$

**Update Rule:**

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

where:

- $f_t$ = forget gate (what to forget)
- $i_t$ = input gate (what to add)
- $\tilde{c}_t$ = candidate values (new information)
- $\odot$ = element-wise multiplication (Hadamard product)

**Key Properties:**

- Flows through the entire sequence: $c_0 \rightarrow c_1 \rightarrow c_2 \rightarrow \cdots \rightarrow c_T$
- Modified by gates at each timestep
- Same dimensionality as hidden state

## LSTM Gates

All gates use sigmoid activation: $\sigma(x) = \frac{1}{1 + e^{-x}} \in [0, 1]$

### Forget Gate ($f_t$)

**Purpose:** Decide what information to forget from the cell state.

**Mathematical Formula:**

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

where:

- $W_f \in \mathbb{R}^{d \times (d + d_{in})}$ = forget gate weight matrix
- $b_f \in \mathbb{R}^d$ = forget gate bias vector
- $[h_{t-1}, x_t]$ = concatenation of previous hidden state and current input
- $d_{in}$ = input dimension (1 in your code: $\text{num\_features} \times 1$)

**Output Interpretation:**

$$f_t[i] = \begin{cases}
\text{close to 1} & \text{keep information in } c_{t-1}[i] \\
\text{close to 0} & \text{forget information in } c_{t-1}[i]
\end{cases}$$

**In Your Code:**

- **Layer 1:** $f_t^{(1)} \in \mathbb{R}^{64}$, processes each feature sequentially
- **Layer 2:** $f_t^{(2)} \in \mathbb{R}^{32}$, processes hidden states from Layer 1

**Example:**

$$f_t = [0.9, 0.1, 0.8, 0.95, \ldots]$$

- Dimension 1: Keep 90% of old cell state
- Dimension 2: Keep 10% of old cell state (forget 90%)
- Dimension 3: Keep 80% of old cell state
- Dimension 4: Keep 95% of old cell state

### Input Gate ($i_t$) and Candidate Values ($\tilde{c}_t$)

**Purpose:** Decide what new information to store in the cell state.

**Input Gate:**

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

**Candidate Values:**

$$\tilde{c}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

where:

- $W_i \in \mathbb{R}^{d \times (d + d_{in})}$ = input gate weight matrix
- $W_C \in \mathbb{R}^{d \times (d + d_{in})}$ = candidate weight matrix
- $b_i, b_C \in \mathbb{R}^d$ = bias vectors
- $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \in [-1, 1]$ = hyperbolic tangent

**Interpretation:**

- $i_t[i]$ = how much of candidate value $\tilde{c}_t[i]$ to add
- $\tilde{c}_t[i]$ = new information value (between -1 and 1)

**Cell State Update:**

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

This combines:

1. **Forgetting:** $f_t \odot c_{t-1}$ (what to keep from old state)
2. **Adding:** $i_t \odot \tilde{c}_t$ (what new information to add)

**Example:**

$$\begin{aligned}
c_{t-1} &= [0.5, -0.3, 0.8, \ldots] \\
f_t &= [0.9, 0.2, 0.7, \ldots] \\
i_t &= [0.6, 0.8, 0.3, \ldots] \\
\tilde{c}_t &= [0.4, -0.5, 0.2, \ldots] \\
c_t &= [0.9, 0.2, 0.7, \ldots] \odot [0.5, -0.3, 0.8, \ldots] + [0.6, 0.8, 0.3, \ldots] \odot [0.4, -0.5, 0.2, \ldots] \\
&= [0.45, -0.06, 0.56, \ldots] + [0.24, -0.40, 0.06, \ldots] \\
&= [0.69, -0.46, 0.62, \ldots]
\end{aligned}$$

### Output Gate ($o_t$)

**Purpose:** Decide what parts of the cell state to output as the hidden state.

**Mathematical Formula:**

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

where:

- $W_o \in \mathbb{R}^{d \times (d + d_{in})}$ = output gate weight matrix
- $b_o \in \mathbb{R}^d$ = output gate bias vector

**Hidden State Computation:**

$$h_t = o_t \odot \tanh(c_t)$$

**Interpretation:**

- $\tanh(c_t)$ = filtered cell state (scaled to $[-1, 1]$)
- $o_t$ = filter to decide what to output
- $h_t$ = final hidden state (what the LSTM "remembers" for next timestep)

**Example:**

$$\begin{aligned}
c_t &= [0.69, -0.46, 0.62, \ldots] \\
\tanh(c_t) &= [0.60, -0.43, 0.55, \ldots] \\
o_t &= [0.8, 0.3, 0.9, \ldots] \\
h_t &= [0.8, 0.3, 0.9, \ldots] \odot [0.60, -0.43, 0.55, \ldots] \\
&= [0.48, -0.13, 0.50, \ldots]
\end{aligned}$$

## Complete LSTM Computation at Timestep $t$

### Step-by-Step Algorithm

For each timestep $t$ (each feature in your code):

**Input:**

- $x_t$ = current feature value (scalar in your case: $x_t \in \mathbb{R}$)
- $h_{t-1}$ = previous hidden state ($h_{t-1} \in \mathbb{R}^d$)
- $c_{t-1}$ = previous cell state ($c_{t-1} \in \mathbb{R}^d$)

**Step 1: Concatenate Input**

$$z_t = [h_{t-1}, x_t] \in \mathbb{R}^{d + d_{in}}$$

**Step 2: Compute Forget Gate**

$$f_t = \sigma(W_f \cdot z_t + b_f)$$

**Step 3: Compute Input Gate**

$$i_t = \sigma(W_i \cdot z_t + b_i)$$

**Step 4: Compute Candidate Values**

$$\tilde{c}_t = \tanh(W_C \cdot z_t + b_C)$$

**Step 5: Update Cell State**

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**Step 6: Compute Output Gate**

$$o_t = \sigma(W_o \cdot z_t + b_o)$$

**Step 7: Compute Hidden State**

$$h_t = o_t \odot \tanh(c_t)$$

**Output:**

- $h_t$ = new hidden state
- $c_t$ = new cell state

### Matrix Formulation

All gates can be computed simultaneously:

$$\begin{bmatrix}
f_t \\
i_t \\
o_t \\
\tilde{c}_t
\end{bmatrix}
=
\begin{bmatrix}
\sigma(W_f \cdot z_t + b_f) \\
\sigma(W_i \cdot z_t + b_i) \\
\sigma(W_o \cdot z_t + b_o) \\
\tanh(W_C \cdot z_t + b_C)
\end{bmatrix}$$

Then:

$$\begin{aligned}
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}$$

## In Your Specific Code

### Layer 1: LSTM(64)

**Input Shape:** $(N, T, 1)$ where:

- $N$ = batch size
- $T$ = num_features (typically 20)
- $1$ = feature dimension

**Processing:**

$$\begin{aligned}
\text{For } t = 1, 2, \ldots, T: \\
&\quad x_t \in \mathbb{R} \quad \text{(feature } t \text{ value)} \\
&\quad h_{t-1}^{(1)} \in \mathbb{R}^{64} \quad \text{(previous hidden state)} \\
&\quad c_{t-1}^{(1)} \in \mathbb{R}^{64} \quad \text{(previous cell state)} \\
&\quad z_t = [h_{t-1}^{(1)}, x_t] \in \mathbb{R}^{65} \\
&\quad \text{Compute gates: } f_t^{(1)}, i_t^{(1)}, o_t^{(1)}, \tilde{c}_t^{(1)} \in \mathbb{R}^{64} \\
&\quad c_t^{(1)} = f_t^{(1)} \odot c_{t-1}^{(1)} + i_t^{(1)} \odot \tilde{c}_t^{(1)} \\
&\quad h_t^{(1)} = o_t^{(1)} \odot \tanh(c_t^{(1)})
\end{aligned}$$

**Output:** Since $\text{return\_sequences=True}$:

$$H^{(1)} = [h_1^{(1)}, h_2^{(1)}, \ldots, h_T^{(1)}] \in \mathbb{R}^{T \times 64}$$

### Layer 2: LSTM(32)

**Input:** $H^{(1)} \in \mathbb{R}^{T \times 64}$ (sequence from Layer 1)

**Processing:**

$$\begin{aligned}
\text{For } t = 1, 2, \ldots, T: \\
&\quad h_t^{(1)} \in \mathbb{R}^{64} \quad \text{(hidden state from Layer 1)} \\
&\quad h_{t-1}^{(2)} \in \mathbb{R}^{32} \quad \text{(previous hidden state)} \\
&\quad c_{t-1}^{(2)} \in \mathbb{R}^{32} \quad \text{(previous cell state)} \\
&\quad z_t = [h_{t-1}^{(2)}, h_t^{(1)}] \in \mathbb{R}^{96} \quad \text{(concatenate 32 + 64)} \\
&\quad \text{Compute gates: } f_t^{(2)}, i_t^{(2)}, o_t^{(2)}, \tilde{c}_t^{(2)} \in \mathbb{R}^{32} \\
&\quad c_t^{(2)} = f_t^{(2)} \odot c_{t-1}^{(2)} + i_t^{(2)} \odot \tilde{c}_t^{(2)} \\
&\quad h_t^{(2)} = o_t^{(2)} \odot \tanh(c_t^{(2)})
\end{aligned}$$

**Output:** Since $\text{return\_sequences=False}$:

$$h_T^{(2)} \in \mathbb{R}^{32} \quad \text{(only final hidden state)}$$

### Final Classification

$$\begin{aligned}
h_T^{(2)} &\in \mathbb{R}^{32} \quad \text{(final hidden state)} \\
\hat{y} &= \text{softmax}(W_{out} \cdot h_T^{(2)} + b_{out}) \\
\hat{y} &\in \mathbb{R}^{C} \quad \text{(class probabilities)}
\end{aligned}$$

where $C = \text{num\_classes}$ (2 for binary classification).

## Parameter Count

### Layer 1: LSTM(64)

For each gate, weight matrix size: $W \in \mathbb{R}^{64 \times 65}$ (64 units, 65 inputs: 64 from $h_{t-1}$ + 1 from $x_t$)

$$\begin{aligned}
\text{Forget Gate:} &\quad W_f: 64 \times 65 = 4,160 \text{ params}, \quad b_f: 64 \text{ params} \\
\text{Input Gate:} &\quad W_i: 64 \times 65 = 4,160 \text{ params}, \quad b_i: 64 \text{ params} \\
\text{Output Gate:} &\quad W_o: 64 \times 65 = 4,160 \text{ params}, \quad b_o: 64 \text{ params} \\
\text{Candidate:} &\quad W_C: 64 \times 65 = 4,160 \text{ params}, \quad b_C: 64 \text{ params} \\
\text{Total Layer 1:} &\quad 4 \times (4,160 + 64) = 16,896 \text{ parameters}
\end{aligned}$$

### Layer 2: LSTM(32)

Weight matrix size: $W \in \mathbb{R}^{32 \times 96}$ (32 units, 96 inputs: 32 from $h_{t-1}^{(2)}$ + 64 from $h_t^{(1)}$)

$$\begin{aligned}
\text{Forget Gate:} &\quad W_f: 32 \times 96 = 3,072 \text{ params}, \quad b_f: 32 \text{ params} \\
\text{Input Gate:} &\quad W_i: 32 \times 96 = 3,072 \text{ params}, \quad b_i: 32 \text{ params} \\
\text{Output Gate:} &\quad W_o: 32 \times 96 = 3,072 \text{ params}, \quad b_o: 32 \text{ params} \\
\text{Candidate:} &\quad W_C: 32 \times 96 = 3,072 \text{ params}, \quad b_C: 32 \text{ params} \\
\text{Total Layer 2:} &\quad 4 \times (3,072 + 32) = 12,416 \text{ parameters}
\end{aligned}$$

### Output Layer

$$\text{Dense: } 32 \times C + C \text{ parameters}$$

### Total Parameters

$$\text{Total} = 16,896 + 12,416 + (32C + C) = 29,312 + 33C$$

For binary classification ($C = 2$): $\text{Total} = 29,378$ parameters

## Key Differences: Hidden State vs Cell State

| **Aspect** | **Hidden State ($h_t$)** | **Cell State ($c_t$)** |
|------------|--------------------------|------------------------|
| Purpose | Short-term memory, output | Long-term memory, storage |
| Visibility | Exposed to next layer | Internal to LSTM |
| Update | Filtered cell state | Direct gate operations |
| Range | $[-1, 1]$ (via $\tanh$) | Unbounded (before $\tanh$) |
| Output | Used for predictions | Used internally |

## Summary

The LSTM processes your tabular features sequentially:

1. Each feature becomes a timestep
2. At each timestep, gates control information flow
3. Cell state stores long-term information
4. Hidden state represents what to output
5. Final hidden state goes to classification layer

All gate computations are handled automatically by TensorFlow's `LSTM()` layer—you don't need to implement them manually.

