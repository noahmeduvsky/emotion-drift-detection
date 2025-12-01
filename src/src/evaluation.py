"""
Evaluation utilities. Computes metrics for classification and drift detection.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, desc=None: x


def compute_metrics(predictions: np.ndarray,
                   labels: np.ndarray,
                   emotion_classes: Optional[List[str]] = None) -> Dict:
    """
    Computes classification metrics. Returns accuracy, F1, precision, recall, etc.
    """
    # convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # convert lists to arrays
    if isinstance(predictions, list):
        predictions = np.array(predictions)
    if isinstance(labels, list):
        labels = np.array(labels)
    
    # flatten everything
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # compute the metrics
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    precision = precision_score(labels, predictions, average='macro', zero_division=0)
    recall = recall_score(labels, predictions, average='macro', zero_division=0)
    
    # get per-class F1 scores
    per_class_f1 = f1_score(labels, predictions, average=None, zero_division=0)
    
    metrics = {
        'accuracy': float(accuracy),
        'f1_score': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'precision': float(precision),
        'recall': float(recall),
        'per_class_f1': per_class_f1.tolist()
    }
    
    # add per-class metrics if we have class names
    if emotion_classes is not None and len(emotion_classes) == len(per_class_f1):
        for i, emotion in enumerate(emotion_classes):
            metrics[f'f1_{emotion}'] = float(per_class_f1[i])
    
    return metrics


def compute_confusion_matrix(predictions: np.ndarray,
                           labels: np.ndarray,
                           emotion_classes: Optional[List[str]] = None) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        predictions: Array of predicted emotion indices
        labels: Array of true emotion indices
        emotion_classes: Optional list of emotion class names
        
    Returns:
        Confusion matrix as numpy array
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    cm = confusion_matrix(labels, predictions)
    return cm


def detect_emotion_drift(emotion_sequence: np.ndarray,
                        threshold: float = 0.5) -> Tuple[np.ndarray, Dict]:
    """
    Detects emotion drift in a sequence. Returns drift scores and some stats.
    """
    if len(emotion_sequence) < 2:
        return np.array([]), {
            'mean_drift': 0.0,
            'max_drift': 0.0,
            'drift_count': 0,
            'total_transitions': 0
        }
    
    # calculate drift as the difference between consecutive emotions
    drift_scores = np.abs(np.diff(emotion_sequence))
    
    # find significant drifts (above threshold)
    significant_drifts = drift_scores > threshold
    
    stats = {
        'mean_drift': float(np.mean(drift_scores)),
        'max_drift': float(np.max(drift_scores)),
        'drift_count': int(np.sum(significant_drifts)),
        'total_transitions': len(drift_scores),
        'drift_rate': float(np.sum(significant_drifts) / len(drift_scores)) if len(drift_scores) > 0 else 0.0
    }
    
    return drift_scores, stats


def compute_drift_detection_metrics(predicted_sequences: List[np.ndarray],
                                   true_sequences: List[np.ndarray],
                                   threshold: float = 0.5) -> Dict:
    """
    Compute metrics for drift detection accuracy.
    
    Args:
        predicted_sequences: List of predicted emotion sequences
        true_sequences: List of true emotion sequences
        threshold: Threshold for detecting significant drift
        
    Returns:
        Dictionary with drift detection metrics
    """
    all_pred_drifts = []
    all_true_drifts = []
    
    for pred_seq, true_seq in zip(predicted_sequences, true_sequences):
        # Detect drifts in predictions
        pred_drift_scores, _ = detect_emotion_drift(pred_seq, threshold)
        pred_drifts = (pred_drift_scores > threshold).astype(int)
        
        # Detect drifts in true labels
        true_drift_scores, _ = detect_emotion_drift(true_seq, threshold)
        true_drifts = (true_drift_scores > threshold).astype(int)
        
        # Pad to same length
        min_len = min(len(pred_drifts), len(true_drifts))
        all_pred_drifts.extend(pred_drifts[:min_len])
        all_true_drifts.extend(true_drifts[:min_len])
    
    # Convert to arrays
    all_pred_drifts = np.array(all_pred_drifts)
    all_true_drifts = np.array(all_true_drifts)
    
    # Compute binary classification metrics for drift detection
    if len(all_pred_drifts) > 0 and len(all_true_drifts) > 0:
        drift_precision = precision_score(all_true_drifts, all_pred_drifts, zero_division=0)
        drift_recall = recall_score(all_true_drifts, all_pred_drifts, zero_division=0)
        drift_f1 = f1_score(all_true_drifts, all_pred_drifts, zero_division=0)
        drift_accuracy = accuracy_score(all_true_drifts, all_pred_drifts)
    else:
        drift_precision = drift_recall = drift_f1 = drift_accuracy = 0.0
    
    return {
        'drift_precision': float(drift_precision),
        'drift_recall': float(drift_recall),
        'drift_f1': float(drift_f1),
        'drift_accuracy': float(drift_accuracy)
    }


def compute_emotion_transition_matrix(sequences: List[np.ndarray],
                                     emotion_classes: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compute emotion transition matrix.
    
    Args:
        sequences: List of emotion sequences
        emotion_classes: Optional list of emotion class names
        
    Returns:
        DataFrame with transition counts
    """
    transitions = []
    
    for seq in sequences:
        for i in range(len(seq) - 1):
            transitions.append((seq[i], seq[i+1]))
    
    if len(transitions) == 0:
        return pd.DataFrame()
    
    transition_df = pd.DataFrame(transitions, columns=['from', 'to'])
    transition_matrix = pd.crosstab(transition_df['from'], transition_df['to'])
    
    # Normalize to probabilities
    transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0)
    
    # Add row/column labels if emotion classes provided
    if emotion_classes is not None:
        transition_matrix.index = [emotion_classes[i] if i < len(emotion_classes) else str(i) 
                                   for i in transition_matrix.index]
        transition_matrix.columns = [emotion_classes[i] if i < len(emotion_classes) else str(i) 
                                     for i in transition_matrix.columns]
    
    return transition_matrix


def compute_trajectory_metrics(sequences: List[np.ndarray],
                              emotion_classes: Optional[List[str]] = None) -> Dict:
    """
    Compute trajectory analysis metrics.
    
    Args:
        sequences: List of emotion sequences
        emotion_classes: Optional list of emotion class names
        
    Returns:
        Dictionary with trajectory metrics
    """
    stability_scores = []
    drift_correlations = []
    
    for seq in sequences:
        if len(seq) < 2:
            continue
        
        # Emotion stability: variance of emotions in sequence
        stability = 1.0 / (1.0 + np.var(seq))
        stability_scores.append(stability)
        
        # Drift correlation: correlation between consecutive emotion changes
        if len(seq) > 2:
            first_half = seq[:len(seq)//2]
            second_half = seq[len(seq)//2:]
            if len(first_half) > 1 and len(second_half) > 1:
                first_drift = np.diff(first_half)
                second_drift = np.diff(second_half)
                if len(first_drift) > 0 and len(second_drift) > 0:
                    min_len = min(len(first_drift), len(second_drift))
                    corr = np.corrcoef(first_drift[:min_len], second_drift[:min_len])[0, 1]
                    if not np.isnan(corr):
                        drift_correlations.append(corr)
    
    metrics = {
        'mean_stability': float(np.mean(stability_scores)) if stability_scores else 0.0,
        'std_stability': float(np.std(stability_scores)) if stability_scores else 0.0,
        'mean_drift_correlation': float(np.mean(drift_correlations)) if drift_correlations else 0.0
    }
    
    return metrics


def evaluate_model(model: torch.nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  device: torch.device,
                  emotion_classes: Optional[List[str]] = None) -> Dict:
    """
    Evaluates a trained model. Runs it on the dataloader and computes all the metrics.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_sequences_pred = []
    all_sequences_true = []
    
    # add progress bar if available
    try:
        from tqdm import tqdm
        progress_bar = tqdm(dataloader, desc="Evaluating")
    except ImportError:
        progress_bar = dataloader
    
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            
            # forward pass
            if hasattr(model.base_model, 'transformer'):
                # reshape for transformer model
                batch_size, seq_len, turn_len = input_ids.shape
                input_ids_flat = input_ids.view(batch_size * seq_len, turn_len)
                attention_mask_flat = attention_mask.view(batch_size * seq_len, turn_len)
                
                logits_flat = model(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
                # use [CLS] token for each turn
                num_classes = logits_flat.shape[-1]
                logits = logits_flat[:, 0, :].view(batch_size, seq_len, num_classes)
            else:
                raise NotImplementedError("LSTM evaluation requires pre-computed embeddings")
            
            # get predictions
            predictions = torch.argmax(logits, dim=-1)
            
            # collect predictions and labels (only valid positions, ignore padding)
            for i in range(predictions.shape[0]):
                valid_mask = padding_mask[i].cpu().numpy()
                pred_seq = predictions[i][valid_mask].cpu().numpy()
                true_seq = labels[i][valid_mask].cpu().numpy()
                
                all_predictions.extend(pred_seq)
                all_labels.extend(true_seq)
                all_sequences_pred.append(pred_seq)
                all_sequences_true.append(true_seq)
    
    # convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # compute all the metrics
    classification_metrics = compute_metrics(all_predictions, all_labels, emotion_classes)
    cm = compute_confusion_matrix(all_predictions, all_labels, emotion_classes)
    drift_metrics = compute_drift_detection_metrics(all_sequences_pred, all_sequences_true)
    trajectory_metrics = compute_trajectory_metrics(all_sequences_true, emotion_classes)
    transition_matrix = compute_emotion_transition_matrix(all_sequences_true, emotion_classes)
    
    # combine everything
    results = {
        'classification': classification_metrics,
        'drift_detection': drift_metrics,
        'trajectory': trajectory_metrics,
        'confusion_matrix': cm.tolist(),
        'transition_matrix': transition_matrix.to_dict() if not transition_matrix.empty else {}
    }
    
    return results

