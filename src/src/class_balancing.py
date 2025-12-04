"""
Class balancing utilities. Handles imbalanced datasets with weighted loss and focal loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List
from sklearn.utils.class_weight import compute_class_weight


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced datasets. Focuses on hard examples and down-weights easy ones.
    Adapted from Lin et al. (2017) for emotion classification tasks.
    """
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, 
                 gamma: float = 2.0, 
                 reduction: str = 'mean'):
        """
        Sets up focal loss. gamma controls how much to focus on hard examples.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes focal loss. The formula is a bit complex but it works well for imbalanced data.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def compute_class_weights_from_labels(labels: List[int] or np.ndarray,
                                      num_classes: Optional[int] = None,
                                      method: str = 'balanced') -> torch.Tensor:
    """
    Computes class weights from labels. Helps balance the loss when classes are imbalanced.
    """
    labels = np.array(labels)
    
    if num_classes is None:
        num_classes = len(np.unique(labels))
    
    if method == 'balanced':
        # use sklearn's balanced weights (inverse frequency)
        weights = compute_class_weight(
            'balanced',
            classes=np.arange(num_classes),
            y=labels
        )
    elif method == 'inverse':
        # simple inverse frequency method
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        weights = np.zeros(num_classes)
        for cls, count in zip(unique, counts):
            weights[cls] = total / (num_classes * count)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return torch.tensor(weights, dtype=torch.float32)


def compute_class_weights_from_emotion_arrays(emotions: List[np.ndarray],
                                               num_classes: int,
                                               method: str = 'balanced') -> torch.Tensor:
    """
    Computes class weights from emotion arrays. Flattens everything first.
    """
    # flatten all the emotion labels
    all_labels = []
    for emotion_array in emotions:
        all_labels.extend(emotion_array.flatten().tolist())
    
    # remove padding values
    all_labels = [l for l in all_labels if l >= 0]
    
    return compute_class_weights_from_labels(all_labels, num_classes, method)


def get_class_weights_for_dataloader(dataloader,
                                     num_classes: int,
                                     method: str = 'balanced') -> torch.Tensor:
    """
    Compute class weights by iterating through a dataloader.
    
    Args:
        dataloader: PyTorch DataLoader
        num_classes: Number of classes
        method: Weighting method
    
    Returns:
        Tensor of class weights
    """
    all_labels = []
    
    for batch in dataloader:
        labels = batch['labels']
        padding_mask = batch.get('padding_mask', None)
        
        if padding_mask is not None:
            valid_labels = labels[padding_mask.bool()]
            all_labels.extend(valid_labels.cpu().numpy().tolist())
        else:
            all_labels.extend(labels.flatten().cpu().numpy().tolist())
    
    # Remove padding values
    all_labels = [l for l in all_labels if l >= 0]
    
    return compute_class_weights_from_labels(all_labels, num_classes, method)


def create_weighted_loss_function(class_weights: Optional[torch.Tensor] = None,
                                  loss_type: str = 'cross_entropy',
                                  focal_gamma: float = 2.0,
                                  device: Optional[torch.device] = None) -> nn.Module:
    """
    Creates a loss function with class balancing. Can use cross entropy or focal loss.
    """
    if class_weights is not None and device is not None:
        class_weights = class_weights.to(device)
    
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    elif loss_type == 'focal':
        return FocalLoss(alpha=class_weights, gamma=focal_gamma)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def print_class_distribution(labels: List[int] or np.ndarray,
                             class_names: Optional[List[str]] = None,
                             title: str = "Class Distribution"):
    """
    Print class distribution statistics.
    
    Args:
        labels: Array of class labels
        class_names: Optional list of class names
        title: Title for the output
    """
    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    print(f"\n{title}:")
    print("-" * 60)
    
    for cls, count in zip(unique, counts):
        percentage = 100 * count / total
        class_name = class_names[cls] if class_names else f"Class {cls}"
        print(f"  {class_name:20s}: {count:6d} ({percentage:5.2f}%)")
    
    print("-" * 60)
    print(f"  Total: {total:6d} samples")
    
    # Calculate imbalance ratio
    if len(counts) > 1:
        max_count = counts.max()
        min_count = counts.min()
        imbalance_ratio = max_count / min_count
        print(f"  Imbalance Ratio: {imbalance_ratio:.2f}:1 (max/min)")
    
    print()


def analyze_class_imbalance(emotions: List[np.ndarray],
                            class_names: Optional[List[str]] = None) -> dict:
    """
    Analyze class imbalance in emotion arrays.
    
    Args:
        emotions: List of emotion arrays
        class_names: Optional class names
    
    Returns:
        Dictionary with imbalance statistics
    """
    all_labels = []
    for emotion_array in emotions:
        all_labels.extend(emotion_array.flatten().tolist())
    
    all_labels = np.array([l for l in all_labels if l >= 0])
    
    unique, counts = np.unique(all_labels, return_counts=True)
    total = len(all_labels)
    
    percentages = counts / total * 100
    
    stats = {
        'total_samples': total,
        'num_classes': len(unique),
        'class_counts': dict(zip(unique, counts)),
        'class_percentages': dict(zip(unique, percentages)),
        'max_count': counts.max(),
        'min_count': counts.min(),
        'imbalance_ratio': counts.max() / counts.min() if counts.min() > 0 else float('inf')
    }
    
    print(f"\nClass Imbalance Analysis:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Number of classes: {stats['num_classes']}")
    print(f"  Imbalance ratio: {stats['imbalance_ratio']:.2f}:1")
    print(f"  Most common: {stats['max_count']} samples")
    print(f"  Least common: {stats['min_count']} samples")
    
    if class_names:
        print(f"\nPer-class distribution:")
        for cls in unique:
            name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
            count = stats['class_counts'][cls]
            pct = stats['class_percentages'][cls]
            print(f"    {name:15s}: {count:6d} ({pct:5.2f}%)")
    
    return stats

