"""
Configuration settings for the Clinical Data Analysis project.
"""

# Data file paths
DATA_PATHS = {
    'asd_train': 'autism_train.csv',
    'asd_test': 'autism_test.csv',
    'id_train': 'intellectual_train.csv',
    'id_test': 'intellectual_test.csv'
}

# Analysis parameters
ANALYSIS_CONFIG = {
    'ceiling_strategy': 'drop',  # Options: 'drop' or 'bin'
    'ceiling_threshold': 50,     # Percentage threshold for ceiling effect detection
    'test_size': 0.2,           # Train/test split ratio
    'random_seed': 42,          # Random seed for reproducibility
    'missing_value_strategy': 'median',  # Strategy for imputing missing values
    'skewness_threshold': 1.0,  # Threshold for applying power transformation
}

# Feature mappings for data harmonization
COLUMN_MAPPING = {
    'Verbal And Intellectual Ability': 'Verbal_Intellectual_Ability',
    'Verbal and Intellectual Ability (ASD)': 'Verbal_Intellectual_Ability',
    'Attention deficit (Intellectual-ASD-ADHD)': 'Attention_Deficit',
    'Sustained Attention (ASD)': 'Sustained_Attention',
    'Empathy (ASD)': 'Empathy',
    'Perception.1': 'Perception_Secondary',
    'Pragmatic Perception': 'Pragmatic_Perception',
    'visual-motor integration': 'Visual_Motor_Integration',
    'Pre-writing': 'Pre_Writing',
    'Fine Motor': 'Fine_Motor',
    'Spatial Orientation': 'Spatial_Orientation',
    'Cognitive Flexibility': 'Cognitive_Flexibility'
}

# Features prone to ceiling effects
CEILING_PRONE_FEATURES = ['Attention_Deficit', 'Visual_Motor_Integration']

# Key features for visualization
KEY_FEATURES_FOR_VIZ = [
    'Verbal_Intellectual_Ability', 
    'Cognitive_Flexibility',
    'Memory', 
    'Sequencing', 
    'Perception'
]

# Plot settings
PLOT_CONFIG = {
    'figsize_default': (10, 6),
    'figsize_comparison': (12, 6),
    'figsize_correlation': (20, 8),
    'dpi': 300,
    'bins': 20,
    'alpha': 0.7
}