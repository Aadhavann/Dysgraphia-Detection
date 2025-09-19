#!/usr/bin/env python3
"""
Main execution script for Enhanced Clinical Data Analysis.

This script runs the complete analysis pipeline for clinical datasets,
including data loading, quality analysis, preprocessing, and visualization.
"""

import os
import sys
import argparse
from datetime import datetime
from config import ANALYSIS_CONFIG, DATA_PATHS
from analyzer import EnhancedClinicalDataAnalyzer


def setup_tensorflow():
    """Setup TensorFlow to avoid warnings and set memory growth."""
    try:
        import tensorflow as tf
        # Set random seed
        tf.random.set_seed(ANALYSIS_CONFIG['random_seed'])
        
        # Configure GPU memory growth if available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Found {len(gpus)} GPU(s), memory growth enabled")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
    except ImportError:
        print("TensorFlow not found, skipping GPU configuration")


def validate_input_files(file_paths):
    """
    Validate that input files exist.
    
    Args:
        file_paths: Dictionary of file paths to validate
        
    Returns:
        bool: True if all files exist, False otherwise
    """
    missing_files = []
    for name, path in file_paths.items():
        if path and not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        print("ERROR: The following files were not found:")
        for file_info in missing_files:
            print(f"  - {file_info}")
        return False
    
    return True


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Enhanced Clinical Data Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Use default file paths
  python main.py --strategy bin                     # Use binning strategy for ceiling effects
  python main.py --asd-train custom_asd_train.csv  # Use custom file path
  python main.py --output-dir results/              # Save results to custom directory
        """
    )
    
    # File path arguments
    parser.add_argument('--asd-train', 
                       default=DATA_PATHS['asd_train'],
                       help='Path to ASD training data CSV file')
    parser.add_argument('--asd-test', 
                       default=DATA_PATHS['asd_test'],
                       help='Path to ASD test data CSV file')
    parser.add_argument('--id-train', 
                       default=DATA_PATHS['id_train'],
                       help='Path to ID training data CSV file')
    parser.add_argument('--id-test', 
                       default=DATA_PATHS['id_test'],
                       help='Path to ID test data CSV file')
    
    # Analysis configuration
    parser.add_argument('--strategy', 
                       choices=['drop', 'bin'],
                       default=ANALYSIS_CONFIG['ceiling_strategy'],
                       help='Strategy for handling ceiling effects')
    parser.add_argument('--threshold', 
                       type=float,
                       default=ANALYSIS_CONFIG['ceiling_threshold'],
                       help='Ceiling effect detection threshold (percentage)')
    
    # Output configuration
    parser.add_argument('--output-dir', 
                       default='.',
                       help='Directory to save output files')
    parser.add_argument('--verbose', '-v', 
                       action='store_true',
                       help='Enable verbose output')
    
    # Analysis options
    parser.add_argument('--skip-eda', 
                       action='store_true',
                       help='Skip exploratory data analysis')
    parser.add_argument('--skip-plots', 
                       action='store_true',
                       help='Skip plot generation')
    
    return parser.parse_args()


def setup_output_directory(output_dir):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Change to output directory
    original_dir = os.getcwd()
    os.chdir(output_dir)
    return original_dir


def print_analysis_header(args):
    """Print analysis header with configuration."""
    print("="*80)
    print("ENHANCED CLINICAL DATA ANALYSIS PIPELINE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Ceiling strategy: {args.strategy}")
    print(f"Ceiling threshold: {args.threshold}%")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print(f"Verbose mode: {'ON' if args.verbose else 'OFF'}")
    print("-"*80)
    
    print("Input files:")
    print(f"  ASD Train: {args.asd_train}")
    print(f"  ASD Test:  {args.asd_test}")
    print(f"  ID Train:  {args.id_train}")
    print(f"  ID Test:   {args.id_test}")
    print("="*80)


def print_analysis_summary(results, elapsed_time):
    """Print final analysis summary."""
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    
    if results['status'] == 'success':
        info = results['final_info']
        print("✓ Analysis completed successfully!")
        print(f"  - Processing time: {elapsed_time:.2f} seconds")
        print(f"  - ASD dataset: {info['asd_shape']} (shape)")
        print(f"  - ID dataset: {info['id_shape']} (shape)")
        print(f"  - Common features: {info['common_features_count']}")
        print(f"  - Ceiling features handled: {len(info['ceiling_features'])}")
        
        print("\nOutput files generated:")
        output_files = [
            'asd_cleaned.csv', 'id_cleaned.csv',
            'ceiling_effects_analysis.csv', 'analysis_summary.csv',
            'enhanced_feature_distributions.png',
            'enhanced_correlation_matrices.png',
            'target_distributions.png',
            'missing_values_heatmap.png'
        ]
        
        for filename in output_files:
            if os.path.exists(filename):
                print(f"  ✓ {filename}")
            else:
                print(f"  - {filename} (not generated)")
        
        # Print train/test splits info
        if 'splits' in results:
            print(f"\nTrain/test splits created: {len(results['splits'])}")
            for split_name in results['splits']:
                print(f"  - {split_name}_cleaned.csv")
    
    else:
        print("✗ Analysis failed!")
        print(f"  Error: {results.get('error_message', 'Unknown error')}")
    
    print("="*80)


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup output directory
    original_dir = setup_output_directory(args.output_dir)
    
    try:
        # Print header
        print_analysis_header(args)
        
        # Setup TensorFlow
        setup_tensorflow()
        
        # Validate input files
        file_paths = {
            'asd_train': args.asd_train,
            'asd_test': args.asd_test,
            'id_train': args.id_train,
            'id_test': args.id_test
        }
        
        if not validate_input_files(file_paths):
            return 1
        
        # Record start time
        start_time = datetime.now()
        
        # Initialize analyzer with custom configuration
        analyzer = EnhancedClinicalDataAnalyzer(ceiling_strategy=args.strategy)
        
        # Update ceiling threshold if specified
        if args.threshold != ANALYSIS_CONFIG['ceiling_threshold']:
            print(f"Using custom ceiling threshold: {args.threshold}%")
        
        # Run complete analysis
        print("\nStarting analysis pipeline...")
        results = analyzer.run_complete_analysis(
            asd_train_path=args.asd_train,
            asd_test_path=args.asd_test,
            id_train_path=args.id_train,
            id_test_path=args.id_test
        )
        
        # Calculate elapsed time
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        # Print summary
        print_analysis_summary(results, elapsed_time)
        
        # Return appropriate exit code
        return 0 if results['status'] == 'success' else 1
    
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user!")
        return 1
    
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    finally:
        # Return to original directory
        os.chdir(original_dir)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)