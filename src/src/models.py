"""
Model architectures for emotion drift detection.
Includes LSTM and Transformer-based sequence models.
"""

import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel, BertConfig, RobertaConfig
from typing import Optional


class EmotionLSTM(nn.Module):
    """
    Bidirectional LSTM model for emotion classification in dialogue sequences.
    """
    
    def __init__(self,
                 embedding_dim: int = 768,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 num_emotions: int = 7,
                 dropout: float = 0.3):
        """
        Initialize LSTM model.
        
        Args:
            embedding_dim: Dimension of input embeddings (BERT output size)
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            num_emotions: Number of emotion classes
            dropout: Dropout rate
        """
        super(EmotionLSTM, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, num_emotions)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM.
        
        Args:
            embeddings: Input embeddings [batch_size, seq_len, embedding_dim]
        
        Returns:
            Emotion logits [batch_size, seq_len, num_emotions]
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(embeddings)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Classification layer
        output = self.fc(lstm_out)
        
        return output


class EmotionTransformer(nn.Module):
    """
    Transformer-based model using BERT/RoBERTa for emotion classification.
    """
    
    def __init__(self,
                 model_name: str = "bert-base-uncased",
                 num_emotions: int = 7,
                 dropout: float = 0.3,
                 freeze_base: bool = False):
        """
        Initialize Transformer model.
        
        Args:
            model_name: Hugging Face model name
            num_emotions: Number of emotion classes
            dropout: Dropout rate
            freeze_base: Whether to freeze base transformer weights
        """
        super(EmotionTransformer, self).__init__()
        
        self.model_name = model_name
        
        # Load base transformer model
        if "roberta" in model_name.lower():
            self.transformer = RobertaModel.from_pretrained(model_name)
            self.config = RobertaConfig.from_pretrained(model_name)
        else:
            self.transformer = BertModel.from_pretrained(model_name)
            self.config = BertConfig.from_pretrained(model_name)
        
        # Freeze base model if specified
        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        # Classification head
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_emotions)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Transformer.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            Emotion logits [batch_size, seq_len, num_emotions]
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use sequence output (token-level representations)
        sequence_output = outputs.last_hidden_state
        
        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        
        # Classification layer
        logits = self.classifier(sequence_output)
        
        return logits


class EmotionDriftDetector(nn.Module):
    """
    Model that detects emotion drift by comparing consecutive emotion predictions.
    Can wrap either LSTM or Transformer base models.
    """
    
    def __init__(self,
                 base_model: nn.Module,
                 num_emotions: int = 7,
                 drift_threshold: float = 0.5):
        """
        Initialize drift detector.
        
        Args:
            base_model: Base emotion classification model (LSTM or Transformer)
            num_emotions: Number of emotion classes
            drift_threshold: Threshold for detecting significant emotion shifts
        """
        super(EmotionDriftDetector, self).__init__()
        
        self.base_model = base_model
        self.num_emotions = num_emotions
        self.drift_threshold = drift_threshold
        
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass through base model.
        
        Args:
            *args, **kwargs: Arguments passed to base model
        
        Returns:
            Emotion logits
        """
        return self.base_model(*args, **kwargs)
    
    def detect_drift(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Detect emotion drift between consecutive turns.
        
        Args:
            logits: Emotion logits [batch_size, seq_len, num_emotions]
        
        Returns:
            Drift scores [batch_size, seq_len-1] indicating magnitude of emotion change
        """
        # Get predicted probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Get predicted emotion indices
        pred_emotions = torch.argmax(probs, dim=-1)
        
        # Calculate drift: difference between consecutive predictions
        drift = torch.abs(pred_emotions[:, 1:] - pred_emotions[:, :-1])
        
        return drift.float()
    
    def compute_drift_statistics(self, logits: torch.Tensor) -> dict:
        """
        Compute statistics about emotion drift in a sequence.
        
        Args:
            logits: Emotion logits [batch_size, seq_len, num_emotions]
        
        Returns:
            Dictionary with drift statistics
        """
        drift_scores = self.detect_drift(logits)
        
        stats = {
            'mean_drift': drift_scores.mean().item(),
            'max_drift': drift_scores.max().item(),
            'drift_count': (drift_scores > self.drift_threshold).sum().item(),
            'total_transitions': drift_scores.numel()
        }
        
        return stats


def create_model(model_type: str = "transformer",
                model_name: str = "bert-base-uncased",
                num_emotions: int = 7,
                **kwargs) -> nn.Module:
    """
    Factory function to create emotion classification models.
    
    Args:
        model_type: Type of model ('lstm' or 'transformer')
        model_name: Hugging Face model name (for transformer)
        num_emotions: Number of emotion classes
        **kwargs: Additional arguments for model initialization
    
    Returns:
        Initialized model
    """
    if model_type.lower() == "lstm":
        embedding_dim = kwargs.get('embedding_dim', 768)
        hidden_dim = kwargs.get('hidden_dim', 256)
        num_layers = kwargs.get('num_layers', 2)
        dropout = kwargs.get('dropout', 0.3)
        
        base_model = EmotionLSTM(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_emotions=num_emotions,
            dropout=dropout
        )
        
    elif model_type.lower() == "transformer":
        dropout = kwargs.get('dropout', 0.3)
        freeze_base = kwargs.get('freeze_base', False)
        
        base_model = EmotionTransformer(
            model_name=model_name,
            num_emotions=num_emotions,
            dropout=dropout,
            freeze_base=freeze_base
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Wrap in drift detector
    drift_threshold = kwargs.get('drift_threshold', 0.5)
    model = EmotionDriftDetector(
        base_model=base_model,
        num_emotions=num_emotions,
        drift_threshold=drift_threshold
    )
    
    return model


if __name__ == "__main__":
    # Example usage
    print("Testing model creation...")
    
    # Create transformer model
    transformer_model = create_model(
        model_type="transformer",
        model_name="bert-base-uncased",
        num_emotions=7
    )
    
    print(f"Transformer model created: {transformer_model}")
    
    # Create LSTM model
    lstm_model = create_model(
        model_type="lstm",
        num_emotions=7,
        embedding_dim=768,
        hidden_dim=256
    )
    
    print(f"LSTM model created: {lstm_model}")

