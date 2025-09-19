"""
Data loading and basic processing utilities.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from config import DATA_PATHS, COLUMN_MAPPING


class DataLoader:
    """Handles loading and basic processing of clinical datasets."""
    
    def __init__(self):
        self.asd_combined = None
        self.id_combined = None
        
    def load_and_combine_data(self, 
                            asd_train_path: str = None,
                            asd_test_path: str = None, 
                            id_train_path: str = None,
                            id_test_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and combine train/test datasets properly.
        
        Args:
            asd_train_path: Path to ASD training data
            asd_test_path: Path to ASD test data
            id_train_path: Path to ID training data
            id_test_path: Path to ID test data
            
        Returns:
            Tuple of (asd_combined, id_combined) DataFrames
        """
        # Use default paths if not provided
        paths = {
            'asd_train': asd_train_path or DATA_PATHS['asd_train'],
            'asd_test': asd_test_path or DATA_PATHS['asd_test'],
            'id_train': id_train_path or DATA_PATHS['id_train'],
            'id_test': id_test_path or DATA_PATHS['id_test']
        }
        
        # Load all datasets
        datasets = {}
        for key, path in paths.items():
            try:
                datasets[key] = pd.read_csv(path)
                print(f"Loaded {key}: {datasets[key].shape}")
            except FileNotFoundError:
                print(f"Warning: Could not find {path}")
                return None, None
        
        # Combine train and test for each condition
        self.asd_combined = pd.concat([datasets['asd_train'], datasets['asd_test']], 
                                     ignore_index=True)
        self.id_combined = pd.concat([datasets['id_train'], datasets['id_test']], 
                                    ignore_index=True)
        
        # Clean identifier columns
        self._clean_identifier_columns()
        
        # Harmonize column names
        self._harmonize_column_names()
        
        # Handle duplicate columns
        self._handle_duplicate_columns()
        
        print(f"ASD Combined Shape: {self.asd_combined.shape}")
        print(f"ID Combined Shape: {self.id_combined.shape}")
        
        return self.asd_combined, self.id_combined
    
    def _clean_identifier_columns(self):
        """Remove identifier columns that aren't needed for analysis."""
        id_columns = ['case_id', 'client_id']
        
        for df in [self.asd_combined, self.id_combined]:
            for col in id_columns:
                if col in df.columns:
                    df.drop(col, axis=1, inplace=True)
    
    def _harmonize_column_names(self):
        """Standardize column names across datasets."""
        for df in [self.asd_combined, self.id_combined]:
            df.rename(columns=COLUMN_MAPPING, inplace=True)
    
    def _handle_duplicate_columns(self):
        """Handle duplicate columns by averaging."""
        # Handle Perception columns specifically
        if ('Perception' in self.asd_combined.columns and 
            'Perception_Secondary' in self.asd_combined.columns):
            
            self.asd_combined['Perception'] = (
                self.asd_combined[['Perception', 'Perception_Secondary']].mean(axis=1)
            )
            self.asd_combined.drop('Perception_Secondary', axis=1, inplace=True)
        
        # Apply same logic to ID dataset if needed
        if ('Perception' in self.id_combined.columns and 
            'Perception_Secondary' in self.id_combined.columns):
            
            self.id_combined['Perception'] = (
                self.id_combined[['Perception', 'Perception_Secondary']].mean(axis=1)
            )
            self.id_combined.drop('Perception_Secondary', axis=1, inplace=True)
    
    def split_labeled_unlabeled(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into labeled and unlabeled subsets.
        
        Args:
            df: DataFrame to split
            
        Returns:
            Tuple of (labeled, unlabeled) DataFrames
        """
        labeled = df[df['target'].isin([0, 1])].copy()
        unlabeled = df[df['target'] == -1].copy()
        return labeled, unlabeled
    
    def get_common_features(self, exclude_target: bool = True) -> list:
        """
        Get features common to both datasets.
        
        Args:
            exclude_target: Whether to exclude 'target' column
            
        Returns:
            List of common feature names
        """
        if self.asd_combined is None or self.id_combined is None:
            return []
        
        common_features = list(
            set(self.asd_combined.columns) & set(self.id_combined.columns)
        )
        
        if exclude_target and 'target' in common_features:
            common_features.remove('target')
            
        return common_features