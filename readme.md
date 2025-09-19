# Enhanced Clinical Data Analysis Pipeline

A comprehensive Python pipeline for analyzing clinical datasets with advanced preprocessing, ceiling effects handling, and exploratory data analysis capabilities.

## Features

- **Automated Data Loading & Harmonization**: Seamlessly combine train/test datasets with intelligent column mapping
- **Ceiling Effects Analysis**: Detect and handle ceiling effects using configurable strategies
- **Advanced Preprocessing**: Missing value imputation, power transformations, and feature scaling
- **Comprehensive EDA**: Automated visualization generation and statistical summaries
- **Flexible Configuration**: Easily customizable analysis parameters
- **Modular Design**: Well-organized codebase with separate components for different analysis stages

## Project Structure

```
clinical-data-analysis/
├── main.py                 # Main execution script
├── config.py              # Configuration settings
├── analyzer.py            # Main analyzer orchestrator
├── data_loader.py         # Data loading utilities
├── ceiling_analysis.py    # Ceiling effects analysis
├── preprocessor.py        # Data preprocessing pipeline
├── visualizer.py          # Visualization and EDA
├── requirements.txt       # Python dependencies
└── README.md             # This file
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

## Troubleshooting

### Common Issues

1. **File Not Found Error**
   ```
   ERROR: The following files were not found:
     - asd_train: autism_train.csv
   ```
   **Solution**: Ensure your CSV files are in the correct location or use custom paths with `--asd-train`, etc.

2. **Memory Issues with Large Datasets**
   **Solution**: The pipeline is optimized for memory efficiency, but for very large datasets, consider processing in chunks.

3. **Missing Dependencies**
   ```
   ImportError: No module named 'pandas'
   ```
   **Solution**: Install requirements with `pip install -r requirements.txt`

### Debugging Tips

- Use `--verbose` flag for detailed output
- Check `analysis_summary.csv` for configuration verification
- Examine individual plots to understand data characteristics

## Contributing

The modular design makes it easy to extend functionality:

1. **Adding New Visualizations**: Extend `visualizer.py`
2. **New Preprocessing Steps**: Add methods to `preprocessor.py`
3. **Custom Analysis Methods**: Create new modules following the existing pattern

## License

This project is provided as-is for educational and research purposes.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the generated `analysis_summary.csv` for configuration details
3. Use `--verbose` mode for detailed debugging output