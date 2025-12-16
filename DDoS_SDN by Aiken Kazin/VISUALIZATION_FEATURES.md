# Visualization Features for Federated Learning

## Overview

This document describes the visualization features added to the Federated Learning pipeline. All visualizations are automatically generated after training completes and saved to the `visualizations/` folder.

## Features

### 1. Class Distribution Heatmaps

**Purpose**: Visualize how data is distributed across workers for both IID and Non-IID scenarios.

**Generated Files**:
- `class_distribution_{MODEL_TYPE}_iid.png`
- `class_distribution_{MODEL_TYPE}_noniid.png`

**What it shows**:
- X-axis: Class labels (Class 0, Class 1)
- Y-axis: Worker IDs (worker1, worker2, ...)
- Color intensity: Number of samples per class per worker

**Example**:
- **IID**: All workers have similar class distributions (~60% Class 0, ~40% Class 1)
- **Non-IID**: Odd workers have 80% Class 0, Even workers have 80% Class 1

### 2. Convergence Plots

**Purpose**: Show how model accuracy and loss change over training rounds.

**Generated Files**:
- `convergence_{MODEL_TYPE}_iid.png`
- `convergence_{MODEL_TYPE}_noniid.png`

**What it shows**:
- **Left subplot**: Accuracy convergence over rounds (percentage)
- **Right subplot**: Loss convergence over rounds
- Both plots include round numbers and metric values annotated on the plot

### 3. Confusion Matrices

**Purpose**: Evaluate model performance by showing true vs predicted labels.

**Generated Files**:
- `confusion_matrix_{MODEL_TYPE}_iid.png`
- `confusion_matrix_{MODEL_TYPE}_noniid.png`

**What it shows**:
- X-axis: Predicted labels
- Y-axis: True labels
- Cell values: Count of samples
- Color intensity: Higher counts = darker blue
- Accuracy percentage displayed in title

## Implementation Details

### Module: `mlops/visualizations.py`

Contains the `FLVisualizer` class with methods:
- `plot_class_distribution_heatmap()`: Generates heatmaps
- `plot_convergence()`: Generates convergence plots
- `plot_confusion_matrix()`: Generates confusion matrices
- `plot_combined_convergence()`: Compares all models (future enhancement)

### Data Collection

**From Workers**:
- Class distribution per worker (sent in `fit()` metrics)
- Predictions and true labels (sent in final `evaluate()` round)

**From Server**:
- Round-by-round accuracy and loss (collected in `aggregate_evaluate()`)
- IID/Non-IID status (extracted from worker metrics)

### Automatic Generation

Visualizations are automatically generated when:
1. Training completes (all rounds finished)
2. Model is saved successfully
3. Final evaluation round completes (for confusion matrix)

## Output Directory Structure

```
visualizations/
├── class_distribution_LSTM_iid.png
├── class_distribution_LSTM_noniid.png
├── class_distribution_CNN1D_iid.png
├── class_distribution_CNN1D_noniid.png
├── convergence_LSTM_iid.png
├── convergence_LSTM_noniid.png
├── confusion_matrix_LSTM_iid.png
└── confusion_matrix_LSTM_noniid.png
```

## Usage

Visualizations are generated automatically. No manual intervention required.

To view visualizations:
```bash
# Navigate to visualizations folder
cd visualizations

# View all generated images
ls -lh *.png
```

## Technical Notes

1. **Image Format**: PNG with 300 DPI resolution
2. **Backend**: Matplotlib with "Agg" backend (non-interactive)
3. **Style**: Seaborn whitegrid style for clean appearance
4. **Memory**: Predictions are only collected on final evaluation round to minimize memory usage

## Future Enhancements

- Combined convergence plots showing all models together
- ROC curves for binary classification
- Precision-Recall curves
- Per-worker accuracy breakdowns
- Training time visualization

