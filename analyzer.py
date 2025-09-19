"""
Main Clinical Data Analyzer that orchestrates all components.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Optional, Dict, Any, List
from config import ANALYSIS_CONFIG, CEILING_PRONE_FEATURES
from data_loader import DataLoader
from ceiling_analysis import CeilingEffectsAnalyzer
from preprocessor import DataPreprocessor
from visualizer import DataVisualizer

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(ANALYSIS_CONFIG['random_seed'])

class EnhancedClinicalDataAnalyzer:
    """
    Main analyzer class that orchestrates the complete clinical data analysis pipeline.
    """
    
    def __init__(self, ceiling_strategy: str = 'drop'):
        """
        Initialize the enhanced clinical data analyzer.
        
        Args:
            ceiling_strategy: Strategy for handling ceiling effects ('drop' or 'bin')
        """
        self.ceiling_strategy = ceiling_strategy
        
        # Initialize components
        self.data_loader = DataLoader()
        self.ceiling_analyzer = CeilingEffectsAnalyzer(strategy=ceiling_strategy)
        self.preprocessor = DataPreprocessor(ceiling_strategy=ceiling_strategy)
        self.visualizer = DataVisualizer()
        
        # Data storage
        self.asd_combined = None
        self.id_combined = None
        self.common_features = []
        
        print(f"Initialized analyzer with ceiling strategy: {ceiling_strategy}")
    
    def load_and_combine_data(self, 
                            asd_train_path: Optional[str] = None,
                            asd_test_path: Optional[str] = None,
                            id_train_path: Optional[str] = None,
                            id_test_path: Optional[str] = None) -> tuple:
        """
        Load and combine train/test datasets.
        
        Args:
            asd_train_path: Path to ASD training data
            asd_test_path: Path to ASD test data
            id_train_path: Path to ID training data
            id_test_path: Path to ID test data
            
        Returns:
            Tuple of (asd_combined, id_combined) DataFrames
        """
        print("=== Loading and Combining Data ===")
        
        self.asd_combined, self.id_combined = self.data_loader.load_and_combine_data(
            asd_train_path, asd_test_path, id_train_path, id_test_path
        )
        
        if self.asd_combined is None or self.id_combined is None:
            raise ValueError("Failed to load datasets. Check file paths.")
        
        # Get initial common features
        self.common_features = self.data_loader.get_common_features()
        
        return self.asd_combined, self.id_combined
    
    def comprehensive_data_quality_analysis(self) -> List[Dict[str, Any]]:
        """
        Perform comprehensive data quality analysis including ceiling effects.
        
        Returns:
            List of statistical summaries
        """
        print("=== Comprehensive Data Quality Analysis ===")
        
        if self.asd_combined is None or self.id_combined is None:
            raise ValueError("Data must be loaded first. Call load_and_combine_data().")
        
        # Perform ceiling effects analysis
        stats = self.ceiling_analyzer.comprehensive_ceiling_analysis(
            self.asd_combined, self.id_combined
        )
        
        # Create comparison plots for ceiling-prone features
        for feature in CEILING_PRONE_FEATURES:
            if (feature in self.asd_combined.columns and 
                feature in self.id_combined.columns):
                self.visualizer.compare_features_across_datasets(
                    self.asd_combined, self.id_combined, feature
                )
        
        return stats
    
    def preprocess_data(self) -> tuple:
        """
        Perform complete data preprocessing pipeline.
        
        Returns:
            Tuple of preprocessed (asd_df, id_df)
        """
        print("=== Data Preprocessing ===")
        
        if self.asd_combined is None or self.id_combined is None:
            raise ValueError("Data must be loaded first. Call load_and_combine_data().")
        
        # Preprocess data
        asd_processed, id_processed = self.preprocessor.preprocess_data(
            self.asd_combined, self.id_combined
        )
        
        # Update common features after preprocessing
        self.common_features = self.preprocessor.common_features
        
        # Update stored data
        self.asd_combined = asd_processed
        self.id_combined = id_processed
        
        return asd_processed, id_processed
    
    def enhanced_eda(self):
        """Perform enhanced exploratory data analysis."""
        print("=== Enhanced Exploratory Data Analysis ===")
        
        if self.asd_combined is None or self.id_combined is None:
            raise ValueError("Data must be loaded first. Call load_and_combine_data().")
        
        # Main EDA
        self.visualizer.enhanced_eda(
            self.asd_combined, self.id_combined, self.common_features
        )
        
        # Additional visualizations
        self.visualizer.create_target_distribution_plot(
            self.asd_combined, self.id_combined
        )
        
        self.visualizer.create_missing_values_heatmap(
            self.asd_combined, self.id_combined, self.common_features
        )
    
    def create_train_test_splits(self) -> Dict[str, pd.DataFrame]:
        """
        Create proper stratified train/test splits.
        
        Returns:
            Dictionary containing all splits
        """
        print("=== Creating Train/Test Splits ===")
        
        if self.asd_combined is None or self.id_combined is None:
            raise ValueError("Data must be preprocessed first.")
        
        splits = self.preprocessor.create_train_test_splits(
            self.asd_combined, self.id_combined
        )
        
        # Print split summary
        print("\nSplit Summary:")
        for split_name, split_data in splits.items():
            print(f"{split_name}: {split_data.shape}")
            if 'target' in split_data.columns:
                target_counts = split_data['target'].value_counts().sort_index()
                print(f"  Target distribution: {target_counts.to_dict()}")
        
        return splits
    
    def save_processed_data(self):
        """Save all processed data to disk."""
        print("=== Saving Processed Data ===")
        
        if self.asd_combined is None or self.id_combined is None:
            raise ValueError("Data must be preprocessed first.")
        
        splits = self.preprocessor.save_preprocessed_data(
            self.asd_combined, self.id_combined
        )
        
        # Save analysis summary
        self._save_analysis_summary()
        
        return splits
    
    def _save_analysis_summary(self):
        """Save analysis summary to file."""
        summary = {
            'ceiling_strategy': self.ceiling_strategy,
            'ceiling_features_identified': self.ceiling_analyzer.ceiling_features,
            'final_common_features': self.common_features,
            'asd_final_shape': self.asd_combined.shape,
            'id_final_shape': self.id_combined.shape,
            'analysis_config': ANALYSIS_CONFIG
        }
        
        # Convert to DataFrame for easier reading
        summary_items = []
        for key, value in summary.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    summary_items.append({'Category': key, 'Parameter': sub_key, 'Value': str(sub_value)})
            else:
                summary_items.append({'Category': 'Main', 'Parameter': key, 'Value': str(value)})
        
        summary_df = pd.DataFrame(summary_items)
        summary_df.to_csv('analysis_summary.csv', index=False)
        print("Analysis summary saved to 'analysis_summary.csv'")
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the current state of data.
        
        Returns:
            Dictionary with data information
        """
        if self.asd_combined is None or self.id_combined is None:
            return {"status": "No data loaded"}
        
        # Get labeled/unlabeled splits
        asd_labeled, asd_unlabeled = self.data_loader.split_labeled_unlabeled(self.asd_combined)
        id_labeled, id_unlabeled = self.data_loader.split_labeled_unlabeled(self.id_combined)
        
        info = {
            'asd_shape': self.asd_combined.shape,
            'id_shape': self.id_combined.shape,
            'common_features_count': len(self.common_features),
            'common_features': self.common_features,
            'asd_labeled_count': len(asd_labeled),
            'asd_unlabeled_count': len(asd_unlabeled),
            'id_labeled_count': len(id_labeled),
            'id_unlabeled_count': len(id_unlabeled),
            'ceiling_features': self.ceiling_analyzer.ceiling_features,
            'ceiling_strategy': self.ceiling_strategy
        }
        
        return info
    
    def run_complete_analysis(self, 
                            asd_train_path: Optional[str] = None,
                            asd_test_path: Optional[str] = None,
                            id_train_path: Optional[str] = None,
                            id_test_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline.
        
        Args:
            asd_train_path: Path to ASD training data
            asd_test_path: Path to ASD test data
            id_train_path: Path to ID training data
            id_test_path: Path to ID test data
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Step 1: Load data
            self.load_and_combine_data(asd_train_path, asd_test_path, 
                                     id_train_path, id_test_path)
            
            # Step 2: Data quality analysis
            quality_stats = self.comprehensive_data_quality_analysis()
            
            # Step 3: Preprocessing
            asd_processed, id_processed = self.preprocess_data()
            
            # Step 4: EDA
            self.enhanced_eda()
            
            # Step 5: Create splits and save
            splits = self.save_processed_data()
            
            # Step 6: Get final info
            final_info = self.get_data_info()
            
            print("\n" + "="*50)
            print("ANALYSIS COMPLETE")
            print("="*50)
            print(f"Ceiling strategy used: {self.ceiling_strategy}")
            print(f"Ceiling features identified: {self.ceiling_analyzer.ceiling_features}")
            print(f"Final common features: {len(self.common_features)}")
            print(f"ASD final shape: {asd_processed.shape}")
            print(f"ID final shape: {id_processed.shape}")
            
            return {
                'quality_stats': quality_stats,
                'splits': splits,
                'final_info': final_info,
                'status': 'success'
            }
            
        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            return {'status': 'error', 'error_message': str(e)}