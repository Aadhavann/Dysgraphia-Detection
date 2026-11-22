# Explainable Dysgraphia Index (EDI)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Published Research

**Image-Based Dysgraphia Detection with Explainability Using CAMs and EDI**

Published in *2026 International Conference on Computing, Communication, Control and Cyber-Physical Systems (I5CPS)*.

[Read on IEEE Xplore](https://ieeexplore.ieee.org/document/11452617/) | [DOI: 10.1109/I5CPS67958.2026.11452617](https://doi.org/10.1109/I5CPS67958.2026.11452617)

## Overview

The Explainable Dysgraphia Index (EDI) bridges the gap between black-box CNN predictions and clinician-interpretable handwriting analysis. By combining Class Activation Mapping (CAM) with traditional handwriting metrics, EDI provides both accurate screening and explainable insights for pediatric dysgraphia detection.

### Key Innovation
- **Novel interpretability approach**: CAM-weighted traditional handwriting features
- **Clinical relevance**: Provides visual explanations that align with occupational therapy assessments
- **Robust methodology**: Subject-wise data splitting prevents leakage, comprehensive validation pipeline

## Architecture

```
Input Image → CNN Classifier → CAM Generation → Feature Weighting → EDI Score
     ↓              ↓              ↓               ↓            ↓
  Handwriting   Prediction    Attention       Weighted      Final
   Sample       Score         Heatmap        Features      Assessment
```

### Core Components

1. **CNN Backbone**: ResNet-18 or EfficientNet-B0 for base classification
2. **CAM Module**: Grad-CAM/Score-CAM for attention visualization
3. **Feature Extractor**: Four handwriting metrics with CAM weighting
4. **EDI Constructor**: Linear combination with learned/equal weights
5. **Evaluation Suite**: Comprehensive validation and visualization

## Quick Start

### Prerequisites

```bash
pip install torch torchvision opencv-python scikit-learn matplotlib seaborn pandas numpy scipy
```

### Dataset Structure

Organize your data as follows:
```
dataset/
├── Low Potential Dysgraphia/
│   ├── LPD(1).jpg
│   ├── LPD(2).jpg
│   └── ...
└── Potential Dysgraphia/
    ├── PD(1).jpg
    ├── PD(2).jpg
    └── ...
```

### Basic Usage

```python
from edi_pipeline import main_pipeline

# Set your dataset path
DATA_DIR = "/path/to/your/dataset"

# Run the complete pipeline
model, features_df, results, cam_stability = main_pipeline(
    DATA_DIR, 
    model_type='resnet18',  # or 'efficientnet_b0'
    epochs=50,
    batch_size=16
)

# Check results
print(f"CAM Stability: {cam_stability:.3f}")
print(f"EDI Improves CNN: {results['CNN_EDI_Fusion']['auc'] > results['CNN_Only']['auc']}")
```

## Features Extracted

### Traditional Handwriting Metrics
1. **Spacing Irregularity**: Standard deviation of inter-character distances
2. **Baseline Wobble**: Variance from fitted text baseline
3. **Character Size Variance**: Bounding box dimension variations
4. **Stroke Width Standard Deviation**: Line thickness consistency

### 
Each metric is recomputed on CAM-weighted image regions to focus on diagnostically relevant areas.

## Results & Outputs

### Generated Files
```
results/
├── training_curves.png          # Loss/accuracy over epochs
├── feature_distributions.png    # Raw vs CAM-weighted features
├── edi_distributions.png        # EDI score distributions
├── correlation_matrix.png       # Feature correlation heatmap
├── case_study_*.png             # Visual explanations with CAM overlays
├── edi_features.csv             # Complete feature dataset
├── trained_model.pth            # Best model weights
└── results_summary.txt          # Performance metrics summary
```

### Performance Metrics
- Accuracy, ROC-AUC, Precision, Recall
- CAM stability IoU scores
- Feature significance analysis
- Clinical case study visualizations

## 🛡️ Quality Assurance

### Case Study Output
![Dysgraphia Detection - CAM Heatmap](images/cam_overlay.png)

- **Original Image**: Raw handwriting sample
- **CAM Heatmap**: Model attention visualization
- **Feature Comparison**: Raw vs CAM-weighted metrics
- **EDI Scores**: Equal vs weighted index values

## Configuration Options

### Model Architecture
```python
# Choose backbone architecture
model_type='resnet18'        # Lightweight, stable
model_type='efficientnet_b0' # More efficient, slightly better accuracy
```

### CAM Method
```python
# In compute_edi_features()
use_gradcam=True   # Faster, gradient-based
use_gradcam=False  # Score-CAM, more stable
```

### Training Parameters
```python
epochs=50          # Training epochs
batch_size=16      # Batch size (adjust for GPU memory)
learning_rate=0.001 # Adam optimizer learning rate
```

## Installation

1. **Clone or download the project files**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data:**
   - Place your CSV files in the project directory
   - Default expected files:
     - `autism_train.csv`
     - `autism_test.csv`
     - `intellectual_train.csv`
     - `intellectual_test.csv`

## Quick Start

### Basic Usage

Run the analysis with default settings:

```bash
python main.py
```

### Advanced Usage

```bash
# Use binning strategy for ceiling effects
python main.py --strategy bin

# Custom file paths
python main.py --asd-train data/my_asd_train.csv --id-test data/my_id_test.csv

# Save results to specific directory
python main.py --output-dir results/experiment_1/

# Custom ceiling threshold
python main.py --threshold 60.0

# Verbose output
python main.py --verbose
```

### Python API Usage

```python
from analyzer import EnhancedClinicalDataAnalyzer

# Initialize analyzer
analyzer = EnhancedClinicalDataAnalyzer(ceiling_strategy='drop')

# Run complete analysis
results = analyzer.run_complete_analysis(
    asd_train_path='autism_train.csv',
    asd_test_path='autism_test.csv',
    id_train_path='intellectual_train.csv',
    id_test_path='intellectual_test.csv'
)

# Or run step by step
analyzer.load_and_combine_data()
analyzer.comprehensive_data_quality_analysis()
analyzer.preprocess_data()
analyzer.enhanced_eda()
analyzer.save_processed_data()
```

## Configuration

### Main Settings (config.py)

- **`ceiling_strategy`**: How to handle ceiling effects (`'drop'` or `'bin'`)
- **`ceiling_threshold`**: Percentage threshold for detecting ceiling effects (default: 50%)
- **`test_size`**: Train/test split ratio (default: 0.2)
- **`random_seed`**: Random seed for reproducibility (default: 42)

### Ceiling Effect Strategies

1. **Drop Strategy (`'drop'`)**: Removes features with significant ceiling effects
2. **Bin Strategy (`'bin'`)**: Converts features into categorical bins (Normal/Elevated/Ceiling)

## Output Files

The analysis generates several output files:

### Processed Data
- `asd_cleaned.csv` - Cleaned ASD dataset
- `id_cleaned.csv` - Cleaned ID dataset
- `*_train_cleaned.csv` - Training splits
- `*_test_cleaned.csv` - Test splits
- `*_unlabeled_cleaned.csv` - Unlabeled data

### Analysis Reports
- `ceiling_effects_analysis.csv` - Detailed ceiling effects statistics
- `analysis_summary.csv` - Complete analysis configuration and results

### Visualizations
- `enhanced_feature_distributions.png` - Feature distribution comparisons
- `enhanced_correlation_matrices.png` - Correlation heatmaps
- `target_distributions.png` - Target variable distributions
- `missing_values_heatmap.png` - Missing data patterns
- `histogram_*_*.png` - Individual feature histograms
- `boxplot_comparison_*.png` - Cross-dataset feature comparisons

## Module Documentation

### analyzer.py - Main Orchestrator
The main class that coordinates all analysis components:
- `EnhancedClinicalDataAnalyzer`: Primary interface for running analyses
- `run_complete_analysis()`: Execute full pipeline
- `get_data_info()`: Get comprehensive data statistics

### data_loader.py - Data Loading
Handles data loading and basic preprocessing:
- `DataLoader`: Load and combine train/test datasets
- Column name harmonization
- Identifier column cleanup

### ceiling_analysis.py - Ceiling Effects
Specialized ceiling effects detection and handling:
- `CeilingEffectsAnalyzer`: Detect and handle ceiling effects
- Statistical analysis of ceiling patterns
- Configurable handling strategies

### preprocessor.py - Data Preprocessing
Comprehensive preprocessing pipeline:
- `DataPreprocessor`: Complete preprocessing workflow
- Missing value imputation
- Power transformations for skewed data
- Train/test split generation

### visualizer.py - Visualization & EDA
Automated visualization and exploratory analysis:
- `DataVisualizer`: Generate comprehensive visualizations
- Feature distribution comparisons
- Correlation analysis
- Missing data visualization

### config.py - Configuration
Centralized configuration management:
- File paths and analysis parameters
- Feature mappings and plot settings
- Easy customization point

## Command Line Options

```
positional arguments: None

optional arguments:
  -h, --help            show this help message and exit
  --asd-train ASD_TRAIN Path to ASD training data CSV file
  --asd-test ASD_TEST   Path to ASD test data CSV file
  --id-train ID_TRAIN   Path to ID training data CSV file
  --id-test ID_TEST     Path to ID test data CSV file
  --strategy {drop,bin} Strategy for handling ceiling effects
  --threshold THRESHOLD Ceiling effect detection threshold (percentage)
  --output-dir OUTPUT_DIR Directory to save output files
  --verbose, -v         Enable verbose output
  --skip-eda            Skip exploratory data analysis
  --skip-plots          Skip plot generation
```

## Data Format Requirements

### Input CSV Structure
- Must contain a `target` column with values: 0, 1, or -1 (unlabeled)
- Feature columns should be numeric
- `case_id` and `client_id` columns are automatically removed if present

### Expected Features
The pipeline handles these common clinical assessment features:
- Verbal and Intellectual Ability
- Attention Deficit measures
- Visual-Motor Integration
- Cognitive Flexibility
- Memory assessments
- And many others...
