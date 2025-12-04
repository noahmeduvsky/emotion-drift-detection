"""
Model architectures for the emotion drift detection project.
Includes both LSTM and Transformer models, with Transformer as the primary architecture.
"""

import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel, BertConfig, RobertaConfig
from typing import Optional


class EmotionLSTM(nn.Module):
    """
    LSTM model for emotion classification. Uses bidirectional architecture to capture context from both directions.
    """
    
    def __init__(self,
                 embedding_dim: int = 768,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 num_emotions: int = 7,
                 dropout: float = 0.3):
        """
        Sets up the LSTM model. embedding_dim is 768 because that's what BERT outputs.
        """
        super(EmotionLSTM, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # bidirectional LSTM to get context from both directions
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # final classification layer (times 2 because bidirectional)
        self.fc = nn.Linear(hidden_dim * 2, num_emotions)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Takes embeddings and returns emotion predictions.
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
    Transformer model using BERT or RoBERTa. Primary architecture used for emotion classification.
    """
    
    def __init__(self,
                 model_name: str = "bert-base-uncased",
                 num_emotions: int = 7,
                 dropout: float = 0.3,
                 freeze_base: bool = False):
        """
        Sets up the transformer model. Can use BERT or RoBERTa from Hugging Face.
        """
        super(EmotionTransformer, self).__init__()
        
        self.model_name = model_name
        
        # load the base model (BERT or RoBERTa)
        if "roberta" in model_name.lower():
            self.transformer = RobertaModel.from_pretrained(model_name)
            self.config = RobertaConfig.from_pretrained(model_name)
        else:
            self.transformer = BertModel.from_pretrained(model_name)
            self.config = BertConfig.from_pretrained(model_name)
        
        # optionally freeze the base model (I usually don't do this)
        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        # add a classification head on top
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_emotions)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Takes tokenized input and returns emotion logits.
        """
        # get the transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # use the sequence output (all token representations)
        sequence_output = outputs.last_hidden_state
        
        # apply dropout for regularization
        sequence_output = self.dropout(sequence_output)
        
        # classify each token position
        logits = self.classifier(sequence_output)
        
        return logits


class EmotionDriftDetector(nn.Module):
    """
    Wrapper that detects emotion drift. Takes a base model and adds drift detection on top.
    """
    
    def __init__(self,
                 base_model: nn.Module,
                 num_emotions: int = 7,
                 drift_threshold: float = 0.5):
        """
        Sets up the drift detector. Wraps around the base model.
        """
        super(EmotionDriftDetector, self).__init__()
        
        self.base_model = base_model
        self.num_emotions = num_emotions
        self.drift_threshold = drift_threshold
        
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass delegates to the base model.
        """
        return self.base_model(*args, **kwargs)
    
    def detect_drift(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Detects emotion drift by comparing consecutive predictions.
        Returns how much the emotion changed between turns.
        """
        # get probabilities and predictions
        probs = torch.softmax(logits, dim=-1)
        pred_emotions = torch.argmax(probs, dim=-1)
        
        # calculate how much emotions changed between consecutive turns
        drift = torch.abs(pred_emotions[:, 1:] - pred_emotions[:, :-1])
        
        return drift.float()
    
    def compute_drift_statistics(self, logits: torch.Tensor) -> dict:
        """
        Computes drift statistics for the sequence, including mean drift, max drift,
        and drift transition counts. Useful for analysis.
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
    Factory function to create emotion detection models.
    Returns an EmotionDriftDetector wrapping the specified base model type.
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
    
    # wrap it in the drift detector
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

