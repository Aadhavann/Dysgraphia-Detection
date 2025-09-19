"""
Data preprocessing pipeline including scaling, transformation, and imputation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
from config import ANALYSIS_CONFIG
from ceiling_analysis import CeilingEffectsAnalyzer


class DataPreprocessor:
    """Handles comprehensive data preprocessing pipeline."""
    
    def __init__(self, ceiling_strategy: str = 'drop'):
        """
        Initialize preprocessor.
        
        Args:
            ceiling_strategy: Strategy for handling ceiling effects
        """
        self.scaler = StandardScaler()
        self.power_transformer = PowerTransformer(method='yeo-johnson')
        self.ceiling_analyzer = CeilingEffectsAnalyzer(strategy=ceiling_strategy)
        self.common_features = []
    
    def preprocess_data(self, 
                       asd_df: pd.DataFrame, 
                       id_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete preprocessing pipeline with ceiling effect handling.
        
        Args:
            asd_df: ASD dataset
            id_df: ID dataset
            
        Returns:
            Tuple of preprocessed (asd_df, id_df)
        """
        print("=== Starting Data Preprocessing ===")
        
        # Make copies to avoid modifying originals
        asd_processed = asd_df.copy()
        id_processed = id_df.copy()
        
        # Identify and handle ceiling effects
        ceiling_features = self.ceiling_analyzer.identify_ceiling_features(
            asd_processed, id_processed, ANALYSIS_CONFIG['ceiling_threshold']
        )
        
        # Apply ceiling strategy to both datasets
        for feature in ceiling_features:
            asd_processed = self.ceiling_analyzer.apply_ceiling_strategy(
                asd_processed, feature
            )
            id_processed = self.ceiling_analyzer.apply_ceiling_strategy(
                id_processed, feature
            )
        
        # Update common features after ceiling handling
        self.common_features = list(
            set(asd_processed.columns) & set(id_processed.columns)
        )
        if 'target' in self.common_features:
            self.common_features.remove('target')
        
        print(f"Common features after ceiling handling: {len(self.common_features)}")
        print(f"Features: {self.common_features}")
        
        # Handle missing values
        asd_processed, id_processed = self._impute_missing_values(
            asd_processed, id_processed
        )
        
        # Apply power transformation to skewed features
        asd_processed, id_processed = self._apply_power_transformation(
            asd_processed, id_processed
        )
        
        return asd_processed, id_processed
    
    def _impute_missing_values(self, 
                              asd_df: pd.DataFrame, 
                              id_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Handle missing values using median imputation."""
        if len(self.common_features) == 0:
            return asd_df, id_df
        
        print("Imputing missing values...")
        imputer = SimpleImputer(strategy=ANALYSIS_CONFIG['missing_value_strategy'])
        
        # Fit and transform each dataset separately
        asd_df[self.common_features] = imputer.fit_transform(asd_df[self.common_features])
        id_df[self.common_features] = imputer.fit_transform(id_df[self.common_features])
        
        return asd_df, id_df
    
    def _apply_power_transformation(self, 
                                   asd_df: pd.DataFrame, 
                                   id_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply power transformation to handle skewed distributions."""
        if len(self.common_features) == 0:
            return asd_df, id_df
        
        print("Applying power transformation to reduce skewness...")
        
        for df, name in [(asd_df, 'ASD'), (id_df, 'ID')]:
            # Check skewness before transformation
            skewness = df[self.common_features].skew()
            highly_skewed = skewness[
                abs(skewness) > ANALYSIS_CONFIG['skewness_threshold']
            ].index.tolist()
            
            if highly_skewed:
                print(f"Transforming highly skewed features in {name}: {highly_skewed}")
                
                # Apply transformation
                transformer = PowerTransformer(method='yeo-johnson')
                df[highly_skewed] = transformer.fit_transform(df[highly_skewed])
        
        return asd_df, id_df
    
    def create_train_test_splits(self, 
                                asd_df: pd.DataFrame, 
                                id_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create proper stratified train/test splits.
        
        Args:
            asd_df: ASD dataset
            id_df: ID dataset
            
        Returns:
            Dictionary containing train/test splits
        """
        splits = {}
        
        # Split labeled and unlabeled data
        asd_labeled = asd_df[asd_df['target'].isin([0, 1])].copy()
        asd_unlabeled = asd_df[asd_df['target'] == -1].copy()
        id_labeled = id_df[id_df['target'].isin([0, 1])].copy()
        id_unlabeled = id_df[id_df['target'] == -1].copy()
        
        # ASD splits
        if len(asd_labeled) > 0:
            asd_train, asd_test = train_test_split(
                asd_labeled, 
                test_size=ANALYSIS_CONFIG['test_size'],
                stratify=asd_labeled['target'],
                random_state=ANALYSIS_CONFIG['random_seed']
            )
            splits['asd_train'] = asd_train
            splits['asd_test'] = asd_test
            
        if len(asd_unlabeled) > 0:
            splits['asd_unlabeled'] = asd_unlabeled
        
        # ID splits
        if len(id_labeled) > 0:
            id_train, id_test = train_test_split(
                id_labeled, 
                test_size=ANALYSIS_CONFIG['test_size'],
                stratify=id_labeled['target'],
                random_state=ANALYSIS_CONFIG['random_seed']
            )
            splits['id_train'] = id_train
            splits['id_test'] = id_test
            
        if len(id_unlabeled) > 0:
            splits['id_unlabeled'] = id_unlabeled
        
        return splits
    
    def save_preprocessed_data(self, 
                              asd_df: pd.DataFrame, 
                              id_df: pd.DataFrame):
        """
        Save preprocessed datasets and splits to disk.
        
        Args:
            asd_df: Preprocessed ASD dataset
            id_df: Preprocessed ID dataset
        """
        # Save main datasets
        asd_df.to_csv('asd_cleaned.csv', index=False)
        id_df.to_csv('id_cleaned.csv', index=False)
        
        # Create and save splits
        splits = self.create_train_test_splits(asd_df, id_df)
        
        for split_name, split_data in splits.items():
            split_data.to_csv(f'{split_name}_cleaned.csv', index=False)
        
        print("Preprocessed datasets saved to disk")
        print(f"Common features ({len(self.common_features)}): {self.common_features}")
        
        return splits