"""
Data visualization and exploratory data analysis utilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
from config import KEY_FEATURES_FOR_VIZ, PLOT_CONFIG


class DataVisualizer:
    """Handles data visualization and exploratory data analysis."""
    
    def __init__(self):
        """Initialize visualizer."""
        # Set default style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def compare_features_across_datasets(self, 
                                       asd_df: pd.DataFrame, 
                                       id_df: pd.DataFrame, 
                                       feature_name: str):
        """
        Compare a feature across ASD and ID datasets with boxplots.
        
        Args:
            asd_df: ASD dataset
            id_df: ID dataset
            feature_name: Feature to compare
        """
        if (feature_name not in asd_df.columns or 
            feature_name not in id_df.columns):
            print(f"Feature {feature_name} not found in both datasets")
            return
        
        # Prepare data for comparison
        asd_data = asd_df[[feature_name, 'target']].copy()
        asd_data['dataset'] = 'ASD'
        
        id_data = id_df[[feature_name, 'target']].copy()
        id_data['dataset'] = 'ID'
        
        combined_data = pd.concat([asd_data, id_data])
        
        # Create comparison plots
        plt.figure(figsize=PLOT_CONFIG['figsize_comparison'])
        
        # By dataset
        plt.subplot(1, 2, 1)
        sns.boxplot(data=combined_data, x='dataset', y=feature_name)
        plt.title(f'{feature_name} by Dataset')
        
        # By target within dataset
        plt.subplot(1, 2, 2)
        labeled_data = combined_data[combined_data['target'].isin([0, 1])]
        if len(labeled_data) > 0:
            sns.boxplot(data=labeled_data, x='dataset', y=feature_name, hue='target')
            plt.title(f'{feature_name} by Dataset and Target')
        
        plt.tight_layout()
        plt.savefig(f'boxplot_comparison_{feature_name.lower()}.png',
                   dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
    
    def enhanced_eda(self, 
                    asd_df: pd.DataFrame, 
                    id_df: pd.DataFrame, 
                    common_features: List[str]):
        """
        Perform enhanced exploratory data analysis.
        
        Args:
            asd_df: ASD dataset
            id_df: ID dataset
            common_features: List of common features
        """
        print("=== Enhanced EDA Report ===")
        
        # Dataset overview
        self._print_dataset_overview(asd_df, id_df)
        
        # Feature distribution comparison
        self._plot_feature_distributions(asd_df, id_df, common_features)
        
        # Correlation analysis
        self._plot_correlation_matrices(asd_df, id_df, common_features)
    
    def _print_dataset_overview(self, asd_df: pd.DataFrame, id_df: pd.DataFrame):
        """Print overview of datasets."""
        for name, df in [('ASD Combined', asd_df), ('ID Combined', id_df)]:
            print(f"\n{name} Overview:")
            print(f"Shape: {df.shape}")
            print("Target distribution:")
            print(df['target'].value_counts().sort_index())
            
            # Split labeled/unlabeled
            labeled = df[df['target'].isin([0, 1])]
            unlabeled = df[df['target'] == -1]
            print(f"Labeled samples: {len(labeled)}")
            print(f"Unlabeled samples: {len(unlabeled)}")
    
    def _plot_feature_distributions(self, 
                                   asd_df: pd.DataFrame, 
                                   id_df: pd.DataFrame, 
                                   common_features: List[str]):
        """Plot feature distributions across datasets."""
        if len(common_features) == 0:
            print("No common features to plot")
            return
        
        # Select features for visualization
        viz_features = [f for f in KEY_FEATURES_FOR_VIZ if f in common_features]
        if not viz_features:
            viz_features = common_features[:6]  # Take first 6 if key ones not available
        
        if not viz_features:
            print("No features available for visualization")
            return
        
        # Setup subplot grid
        n_features = len(viz_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Get labeled data
        asd_labeled = asd_df[asd_df['target'].isin([0, 1])]
        id_labeled = id_df[id_df['target'].isin([0, 1])]
        
        # Plot distributions
        for i, feature in enumerate(viz_features):
            row, col = i // n_cols, i % n_cols
            
            if feature in asd_labeled.columns and feature in id_labeled.columns:
                axes[row, col].hist(asd_labeled[feature].dropna(), 
                                   alpha=PLOT_CONFIG['alpha'],
                                   label='ASD', bins=PLOT_CONFIG['bins'], density=True)
                axes[row, col].hist(id_labeled[feature].dropna(), 
                                   alpha=PLOT_CONFIG['alpha'],
                                   label='ID', bins=PLOT_CONFIG['bins'], density=True)
                axes[row, col].set_title(f'Distribution of {feature}')
                axes[row, col].set_xlabel('Score')
                axes[row, col].set_ylabel('Density')
                axes[row, col].legend()
        
        # Hide empty subplots
        for i in range(n_features, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('enhanced_feature_distributions.png', 
                   dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_matrices(self, 
                                  asd_df: pd.DataFrame, 
                                  id_df: pd.DataFrame, 
                                  common_features: List[str]):
        """Plot correlation matrices for both datasets."""
        if len(common_features) < 2:
            print("Insufficient features for correlation analysis")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=PLOT_CONFIG['figsize_correlation'])
        
        # ASD correlations
        asd_labeled = asd_df[asd_df['target'].isin([0, 1])]
        if len(asd_labeled) > 0 and len(common_features) > 1:
            asd_corr = asd_labeled[common_features].corr()
            sns.heatmap(asd_corr, annot=True, cmap='coolwarm', center=0, 
                       ax=ax1, fmt='.2f', square=True)
            ax1.set_title('ASD Dataset Correlation Matrix')
        
        # ID correlations
        id_labeled = id_df[id_df['target'].isin([0, 1])]
        if len(id_labeled) > 0 and len(common_features) > 1:
            id_corr = id_labeled[common_features].corr()
            sns.heatmap(id_corr, annot=True, cmap='coolwarm', center=0, 
                       ax=ax2, fmt='.2f', square=True)
            ax2.set_title('ID Dataset Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig('enhanced_correlation_matrices.png', 
                   dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
    
    def create_target_distribution_plot(self, 
                                       asd_df: pd.DataFrame, 
                                       id_df: pd.DataFrame):
        """Create visualization of target distributions."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # ASD target distribution
        asd_targets = asd_df['target'].value_counts().sort_index()
        axes[0].bar(asd_targets.index, asd_targets.values, alpha=0.7)
        axes[0].set_title('ASD Dataset Target Distribution')
        axes[0].set_xlabel('Target')
        axes[0].set_ylabel('Count')
        
        # Add count labels on bars
        for i, v in enumerate(asd_targets.values):
            axes[0].text(asd_targets.index[i], v + 0.01 * max(asd_targets.values), 
                        str(v), ha='center', va='bottom')
        
        # ID target distribution
        id_targets = id_df['target'].value_counts().sort_index()
        axes[1].bar(id_targets.index, id_targets.values, alpha=0.7, color='orange')
        axes[1].set_title('ID Dataset Target Distribution')
        axes[1].set_xlabel('Target')
        axes[1].set_ylabel('Count')
        
        # Add count labels on bars
        for i, v in enumerate(id_targets.values):
            axes[1].text(id_targets.index[i], v + 0.01 * max(id_targets.values), 
                        str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('target_distributions.png', 
                   dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
    
    def create_missing_values_heatmap(self, 
                                     asd_df: pd.DataFrame, 
                                     id_df: pd.DataFrame, 
                                     common_features: List[str]):
        """Create heatmap showing missing values pattern."""
        if not common_features:
            print("No common features for missing values analysis")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # ASD missing values
        asd_missing = asd_df[common_features].isnull()
        sns.heatmap(asd_missing, cbar=True, ax=ax1, cmap='viridis')
        ax1.set_title('ASD Dataset Missing Values Pattern')
        ax1.set_ylabel('Samples')
        
        # ID missing values
        id_missing = id_df[common_features].isnull()
        sns.heatmap(id_missing, cbar=True, ax=ax2, cmap='viridis')
        ax2.set_title('ID Dataset Missing Values Pattern')
        ax2.set_ylabel('Samples')
        ax2.set_xlabel('Features')
        
        plt.tight_layout()
        plt.savefig('missing_values_heatmap.png', 
                   dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        
        # Print missing values summary
        print("\n=== Missing Values Summary ===")
        print("ASD Dataset:")
        asd_missing_pct = (asd_df[common_features].isnull().sum() / len(asd_df) * 100)
        missing_features_asd = asd_missing_pct[asd_missing_pct > 0].sort_values(ascending=False)
        if len(missing_features_asd) > 0:
            print(missing_features_asd.to_string())
        else:
            print("No missing values found")
        
        print("\nID Dataset:")
        id_missing_pct = (id_df[common_features].isnull().sum() / len(id_df) * 100)
        missing_features_id = id_missing_pct[id_missing_pct > 0].sort_values(ascending=False)
        if len(missing_features_id) > 0:
            print(missing_features_id.to_string())
        else:
            print("No missing values found")