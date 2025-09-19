"""
Ceiling effects analysis and handling utilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from config import CEILING_PRONE_FEATURES, PLOT_CONFIG


class CeilingEffectsAnalyzer:
    """Analyzes and handles ceiling effects in clinical data."""
    
    def __init__(self, strategy: str = 'drop'):
        """
        Initialize ceiling effects analyzer.
        
        Args:
            strategy: 'drop' to remove ceiling features, 'bin' to categorize them
        """
        self.strategy = strategy
        self.ceiling_features = []
    
    def analyze_ceiling_effects(self, 
                               df: pd.DataFrame, 
                               feature_name: str, 
                               dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Analyze ceiling effects for a specific feature.
        
        Args:
            df: DataFrame containing the feature
            feature_name: Name of the feature to analyze
            dataset_name: Name of dataset for labeling plots
            
        Returns:
            Dictionary with ceiling effects statistics
        """
        if feature_name not in df.columns:
            print(f"Feature {feature_name} not found in {dataset_name} dataset")
            return None
        
        feature_data = df[feature_name].dropna()
        
        # Calculate statistics
        stats = {
            'dataset': dataset_name,
            'feature': feature_name,
            'count': len(feature_data),
            'mean': feature_data.mean(),
            'median': feature_data.median(),
            'std': feature_data.std(),
            'min': feature_data.min(),
            'max': feature_data.max(),
            'count_100': (feature_data == 100).sum(),
            'pct_100': (feature_data == 100).mean() * 100,
            'count_99_100': (feature_data >= 99).sum(),
            'pct_99_100': (feature_data >= 99).mean() * 100
        }
        
        self._print_ceiling_stats(stats)
        self._plot_ceiling_histogram(feature_data, dataset_name, feature_name)
        
        return stats
    
    def _print_ceiling_stats(self, stats: Dict[str, Any]):
        """Print ceiling effects statistics."""
        print(f"\n=== {stats['dataset']} - {stats['feature']} Analysis ===")
        print(f"Count: {stats['count']}")
        print(f"Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}, Std: {stats['std']:.2f}")
        print(f"Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
        print(f"Values = 100: {stats['count_100']} ({stats['pct_100']:.1f}%)")
        print(f"Values ≥ 99: {stats['count_99_100']} ({stats['pct_99_100']:.1f}%)")
    
    def _plot_ceiling_histogram(self, 
                               feature_data: pd.Series, 
                               dataset_name: str, 
                               feature_name: str):
        """Create histogram plot for ceiling effects visualization."""
        plt.figure(figsize=PLOT_CONFIG['figsize_default'])
        plt.hist(feature_data, bins=PLOT_CONFIG['bins'], 
                alpha=PLOT_CONFIG['alpha'], edgecolor='black')
        plt.title(f'{dataset_name} - {feature_name} Distribution')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.axvline(x=100, color='red', linestyle='--', label='Ceiling (100)')
        plt.legend()
        plt.savefig(f'histogram_{dataset_name.lower()}_{feature_name.lower()}.png',
                   dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
    
    def identify_ceiling_features(self, 
                                 asd_df: pd.DataFrame, 
                                 id_df: pd.DataFrame,
                                 threshold: float = 50) -> List[str]:
        """
        Identify features with ceiling effects above threshold percentage.
        
        Args:
            asd_df: ASD dataset
            id_df: ID dataset
            threshold: Percentage threshold for ceiling effect detection
            
        Returns:
            List of features with ceiling effects
        """
        ceiling_features = []
        
        for feature in CEILING_PRONE_FEATURES:
            for df, dataset_name in [(asd_df, 'ASD'), (id_df, 'ID')]:
                if feature in df.columns:
                    pct_ceiling = (df[feature] == 100).mean() * 100
                    if pct_ceiling > threshold:
                        ceiling_features.append(feature)
                        print(f"Ceiling effect detected in {dataset_name} {feature}: "
                              f"{pct_ceiling:.1f}% at maximum")
        
        self.ceiling_features = list(set(ceiling_features))
        return self.ceiling_features
    
    def bin_feature(self, val: float) -> int:
        """
        Bin feature values into categories.
        
        Args:
            val: Feature value
            
        Returns:
            Binned category (0: Normal, 1: Elevated, 2: Ceiling)
        """
        if pd.isna(val):
            return np.nan
        elif val < 71:
            return 0  # Normal
        elif val < 100:
            return 1  # Elevated
        else:
            return 2  # Ceiling
    
    def apply_ceiling_strategy(self, 
                              df: pd.DataFrame, 
                              feature_name: str) -> pd.DataFrame:
        """
        Apply ceiling effect handling strategy to a feature.
        
        Args:
            df: DataFrame to modify
            feature_name: Feature to process
            
        Returns:
            Modified DataFrame
        """
        if feature_name not in df.columns:
            return df
        
        df_copy = df.copy()
        
        if self.strategy == 'drop':
            print(f"Dropping {feature_name} due to ceiling effects")
            return df_copy.drop(feature_name, axis=1)
        
        elif self.strategy == 'bin':
            print(f"Binning {feature_name} due to ceiling effects")
            df_copy[f'{feature_name}_binned'] = df_copy[feature_name].apply(self.bin_feature)
            df_copy = df_copy.drop(feature_name, axis=1)
            return df_copy
        
        return df_copy
    
    def comprehensive_ceiling_analysis(self, 
                                     asd_df: pd.DataFrame, 
                                     id_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Perform comprehensive ceiling effects analysis.
        
        Args:
            asd_df: ASD dataset
            id_df: ID dataset
            
        Returns:
            List of statistical summaries
        """
        all_stats = []
        
        # Analyze each ceiling-prone feature
        for feature in CEILING_PRONE_FEATURES:
            # ASD analysis
            asd_stats = self.analyze_ceiling_effects(asd_df, feature, 'ASD')
            if asd_stats:
                all_stats.append(asd_stats)
            
            # ID analysis
            id_stats = self.analyze_ceiling_effects(id_df, feature, 'ID')
            if id_stats:
                all_stats.append(id_stats)
        
        # Save summary statistics
        if all_stats:
            stats_df = pd.DataFrame(all_stats)
            stats_df.to_csv('ceiling_effects_analysis.csv', index=False)
            print("\nCeiling effects analysis saved to 'ceiling_effects_analysis.csv'")
        
        return all_stats