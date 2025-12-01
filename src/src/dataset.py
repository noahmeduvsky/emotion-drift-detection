"""
PyTorch Dataset and DataLoader utilities for emotion drift detection.
Handles batching of dialogue sequences for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
import pandas as pd


class DialogueDataset(Dataset):
    """
    PyTorch Dataset for dialogue sequences with emotion labels.
    """
    
    def __init__(self, 
                 dialogues: List[Dict],
                 emotions: List[np.ndarray],
                 max_seq_length: Optional[int] = None):
        """
        Initialize dataset.
        
        Args:
            dialogues: List of dialogue dictionaries with 'input_ids' and 'attention_mask'
            emotions: List of numpy arrays with emotion labels for each turn
            max_seq_length: Optional maximum sequence length (for padding/truncation)
        """
        self.dialogues = dialogues
        self.emotions = emotions
        self.max_seq_length = max_seq_length
        
        # Validate that dialogues and emotions match
        assert len(dialogues) == len(emotions), \
            f"Mismatch: {len(dialogues)} dialogues but {len(emotions)} emotion sequences"
    
    def __len__(self) -> int:
        return len(self.dialogues)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single dialogue sequence.
        
        Returns:
            Dictionary with:
            - input_ids: [seq_len, max_turn_length]
            - attention_mask: [seq_len, max_turn_length]
            - labels: [seq_len] emotion labels for each turn
            - dialogue_length: scalar tensor with actual sequence length
        """
        dialogue = self.dialogues[idx]
        emotion_labels = self.emotions[idx]
        
        input_ids = dialogue['input_ids']  # [num_turns, max_turn_length]
        attention_mask = dialogue['attention_mask']  # [num_turns, max_turn_length]
        
        # Convert to tensors if not already
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        # Convert emotion labels to tensor
        if isinstance(emotion_labels, np.ndarray):
            labels = torch.tensor(emotion_labels, dtype=torch.long)
        else:
            labels = torch.tensor(emotion_labels, dtype=torch.long)
        
        # Truncate or pad sequence if max_seq_length is specified
        seq_len = input_ids.shape[0]
        if self.max_seq_length is not None and seq_len > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
            labels = labels[:self.max_seq_length]
            seq_len = self.max_seq_length
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'dialogue_length': torch.tensor(seq_len, dtype=torch.long)
        }


def collate_dialogues(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to batch variable-length dialogue sequences.
    
    Args:
        batch: List of dictionaries from DialogueDataset
        
    Returns:
        Batched tensors with padding:
        - input_ids: [batch_size, max_seq_len, max_turn_length]
        - attention_mask: [batch_size, max_seq_len, max_turn_length]
        - labels: [batch_size, max_seq_len]
        - dialogue_lengths: [batch_size] actual lengths for each dialogue
        - padding_mask: [batch_size, max_seq_len] mask for valid turns
    """
    batch_size = len(batch)
    
    # Find maximum sequence length and turn length in batch
    max_seq_len = max(item['input_ids'].shape[0] for item in batch)
    max_turn_length = max(item['input_ids'].shape[1] for item in batch)
    
    # Initialize batched tensors
    batched_input_ids = torch.zeros(batch_size, max_seq_len, max_turn_length, dtype=torch.long)
    batched_attention_mask = torch.zeros(batch_size, max_seq_len, max_turn_length, dtype=torch.long)
    batched_labels = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    dialogue_lengths = torch.zeros(batch_size, dtype=torch.long)
    padding_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    
    for i, item in enumerate(batch):
        seq_len = item['input_ids'].shape[0]
        turn_len = item['input_ids'].shape[1]
        dialogue_lengths[i] = seq_len
        
        # Copy data (pad turn length if needed)
        batched_input_ids[i, :seq_len, :turn_len] = item['input_ids']
        batched_attention_mask[i, :seq_len, :turn_len] = item['attention_mask']
        batched_labels[i, :seq_len] = item['labels']
        padding_mask[i, :seq_len] = True
    
    return {
        'input_ids': batched_input_ids,
        'attention_mask': batched_attention_mask,
        'labels': batched_labels,
        'dialogue_lengths': dialogue_lengths,
        'padding_mask': padding_mask
    }


def create_dataloader(dialogues: List[Dict],
                      emotions: List[np.ndarray],
                      batch_size: int = 8,
                      shuffle: bool = True,
                      max_seq_length: Optional[int] = None,
                      num_workers: int = 0) -> DataLoader:
    """
    Create a DataLoader for dialogue sequences.
    
    Args:
        dialogues: List of dialogue dictionaries
        emotions: List of emotion label arrays
        batch_size: Batch size
        shuffle: Whether to shuffle data
        max_seq_length: Optional maximum sequence length
        num_workers: Number of worker processes for data loading
        
    Returns:
        DataLoader instance
    """
    dataset = DialogueDataset(dialogues, emotions, max_seq_length=max_seq_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_dialogues,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader


def split_dialogues(dialogues: List[Dict],
                   emotions: List[np.ndarray],
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15,
                   test_ratio: float = 0.15,
                   random_seed: int = 42) -> Tuple[List[Dict], List[np.ndarray], 
                                                    List[Dict], List[np.ndarray],
                                                    List[Dict], List[np.ndarray]]:
    """
    Split dialogues into train/validation/test sets.
    
    Args:
        dialogues: List of dialogue dictionaries
        emotions: List of emotion label arrays
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dialogues, train_emotions, val_dialogues, val_emotions, 
                 test_dialogues, test_emotions)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    np.random.seed(random_seed)
    indices = np.random.permutation(len(dialogues))
    
    n_total = len(dialogues)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train_dialogues = [dialogues[i] for i in train_indices]
    train_emotions = [emotions[i] for i in train_indices]
    
    val_dialogues = [dialogues[i] for i in val_indices]
    val_emotions = [emotions[i] for i in val_indices]
    
    test_dialogues = [dialogues[i] for i in test_indices]
    test_emotions = [emotions[i] for i in test_indices]
    
    return (train_dialogues, train_emotions, 
            val_dialogues, val_emotions,
            test_dialogues, test_emotions)

