"""
Emotion Drift Detection Package
"""

from .data_loader import (
    load_dailydialog_dataset,
    load_emotionlines_dataset,
    load_meld_dataset,
    combine_datasets,
    save_dataset,
    load_local_dataset
)

from .preprocessing import (
    EmotionPreprocessor,
    normalize_emotion_labels,
    balance_dataset
)

from .models import (
    EmotionLSTM,
    EmotionTransformer,
    EmotionDriftDetector,
    create_model
)

__version__ = "0.1.0"

