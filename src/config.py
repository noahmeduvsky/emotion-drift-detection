"""
Configuration file with all the default hyperparameters and settings.
I keep everything here so it's easy to change.
"""

# Model configuration
MODEL_CONFIG = {
    'model_type': 'transformer',  # 'transformer' or 'lstm'
    'model_name': 'bert-base-uncased',  # Hugging Face model name
    'num_emotions': 7,
    'dropout': 0.3,
    'drift_threshold': 0.5
}

# LSTM-specific configuration
LSTM_CONFIG = {
    'embedding_dim': 768,
    'hidden_dim': 256,
    'num_layers': 2
}

# Training configuration
TRAINING_CONFIG = {
    'batch_size': 8,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'num_epochs': 10,
    'early_stopping_patience': 10,
    'max_grad_norm': 1.0
}

# Data configuration
DATA_CONFIG = {
    'dataset': 'dailydialog',
    'max_length': 128,  # Maximum token length per turn
    'max_seq_length': None,  # Maximum dialogue sequence length (None = no limit)
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'random_seed': 42
}

# Preprocessing configuration
PREPROCESSING_CONFIG = {
    'lowercase': True,
    'normalize_emotions': True
}

# Paths
PATHS = {
    'data_dir': 'data',
    'models_dir': 'models',
    'checkpoints_dir': 'models/checkpoints',
    'results_dir': 'results',
    'logs_dir': 'logs'
}

# Emotion classes (standard set)
EMOTION_CLASSES = [
    'joy',
    'anger',
    'sadness',
    'fear',
    'disgust',
    'surprise',
    'neutral'
]

# Sentiment mapping for visualization
SENTIMENT_MAP = {
    'anger': -1.0,
    'sadness': -1.0,
    'fear': -1.0,
    'disgust': -1.0,
    'neutral': 0.0,
    'surprise': 0.5,
    'joy': 1.0
}

