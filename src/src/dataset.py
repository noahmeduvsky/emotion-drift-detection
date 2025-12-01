"""
PyTorch Dataset and DataLoader stuff. Handles batching dialogue sequences for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
import pandas as pd


class DialogueDataset(Dataset):
    """
    PyTorch Dataset for dialogue sequences. Each item is a dialogue with its emotion labels.
    """
    
    def __init__(self, 
                 dialogues: List[Dict],
                 emotions: List[np.ndarray],
                 max_seq_length: Optional[int] = None):
        """
        Sets up the dataset. Can optionally truncate long sequences.
        """
        self.dialogues = dialogues
        self.emotions = emotions
        self.max_seq_length = max_seq_length
        
        # make sure dialogues and emotions match up
        assert len(dialogues) == len(emotions), \
            f"Mismatch: {len(dialogues)} dialogues but {len(emotions)} emotion sequences"
    
    def __len__(self) -> int:
        return len(self.dialogues)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns one dialogue sequence with its labels.
        """
        dialogue = self.dialogues[idx]
        emotion_labels = self.emotions[idx]
        
        input_ids = dialogue['input_ids']
        attention_mask = dialogue['attention_mask']
        
        # convert to tensors if needed
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        # convert labels to tensor
        if isinstance(emotion_labels, np.ndarray):
            labels = torch.tensor(emotion_labels, dtype=torch.long)
        else:
            labels = torch.tensor(emotion_labels, dtype=torch.long)
        
        # truncate if sequence is too long
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
    Custom collate function to batch dialogues. Handles padding since dialogues have different lengths.
    """
    batch_size = len(batch)
    
    # find the max lengths in this batch
    max_seq_len = max(item['input_ids'].shape[0] for item in batch)
    max_turn_length = max(item['input_ids'].shape[1] for item in batch)
    
    # create batched tensors filled with zeros (for padding)
    batched_input_ids = torch.zeros(batch_size, max_seq_len, max_turn_length, dtype=torch.long)
    batched_attention_mask = torch.zeros(batch_size, max_seq_len, max_turn_length, dtype=torch.long)
    batched_labels = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    dialogue_lengths = torch.zeros(batch_size, dtype=torch.long)
    padding_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    
    # copy each dialogue into the batch, padding as needed
    for i, item in enumerate(batch):
        seq_len = item['input_ids'].shape[0]
        turn_len = item['input_ids'].shape[1]
        dialogue_lengths[i] = seq_len
        
        # copy the actual data
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
    Creates a DataLoader for training. Handles all the batching stuff.
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
    Splits dialogues into train/val/test. Uses random seed for reproducibility.
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

