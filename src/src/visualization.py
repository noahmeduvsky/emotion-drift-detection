"""
Visualization utilities. Makes plots for trajectories, confusion matrices, etc.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import os


def plot_emotion_trajectory(emotion_sequence: np.ndarray,
                           emotion_classes: List[str],
                           dialogue_id: Optional[str] = None,
                           save_path: Optional[str] = None,
                           show_plot: bool = True):
    """
    Plot emotion trajectory for a single dialogue.
    
    Args:
        emotion_sequence: Array of emotion indices for each turn
        emotion_classes: List of emotion class names
        dialogue_id: Optional dialogue identifier
        save_path: Optional path to save plot
        show_plot: Whether to display plot
    """
    plt.figure(figsize=(12, 6))
    
    # Map emotions to numeric values for plotting
    turn_ids = np.arange(len(emotion_sequence))
    
    # Plot trajectory
    plt.plot(turn_ids, emotion_sequence, marker='o', linewidth=2, markersize=8)
    
    # Add emotion labels on y-axis
    plt.yticks(range(len(emotion_classes)), emotion_classes)
    plt.xlabel('Turn ID', fontsize=12)
    plt.ylabel('Emotion', fontsize=12)
    
    title = 'Emotion Trajectory'
    if dialogue_id is not None:
        title += f' - Dialogue {dialogue_id}'
    plt.title(title, fontsize=14, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved trajectory plot to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_multiple_trajectories(emotion_sequences: List[np.ndarray],
                              emotion_classes: List[str],
                              dialogue_ids: Optional[List[str]] = None,
                              max_sequences: int = 10,
                              save_path: Optional[str] = None,
                              show_plot: bool = True):
    """
    Plot multiple emotion trajectories on the same plot.
    
    Args:
        emotion_sequences: List of emotion sequences
        emotion_classes: List of emotion class names
        dialogue_ids: Optional list of dialogue identifiers
        max_sequences: Maximum number of sequences to plot
        save_path: Optional path to save plot
        show_plot: Whether to display plot
    """
    plt.figure(figsize=(14, 8))
    
    # Limit number of sequences
    sequences_to_plot = emotion_sequences[:max_sequences]
    if dialogue_ids:
        dialogue_ids_to_plot = dialogue_ids[:max_sequences]
    else:
        dialogue_ids_to_plot = [f'Dialogue {i}' for i in range(len(sequences_to_plot))]
    
    # Plot each trajectory
    for i, (seq, dialogue_id) in enumerate(zip(sequences_to_plot, dialogue_ids_to_plot)):
        turn_ids = np.arange(len(seq))
        plt.plot(turn_ids, seq, marker='o', linewidth=2, markersize=6, 
                label=dialogue_id, alpha=0.7)
    
    # Add emotion labels on y-axis
    plt.yticks(range(len(emotion_classes)), emotion_classes)
    plt.xlabel('Turn ID', fontsize=12)
    plt.ylabel('Emotion', fontsize=12)
    plt.title('Emotion Trajectories Across Multiple Dialogues', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved trajectories plot to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(cm: np.ndarray,
                         emotion_classes: List[str],
                         normalize: bool = True,
                         save_path: Optional[str] = None,
                         show_plot: bool = True):
    """
    Plot confusion matrix for emotion classification.
    
    Args:
        cm: Confusion matrix as numpy array
        emotion_classes: List of emotion class names
        normalize: Whether to normalize the matrix
        save_path: Optional path to save plot
        show_plot: Whether to display plot
    """
    plt.figure(figsize=(10, 8))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=emotion_classes,
                yticklabels=emotion_classes,
                cbar_kws={'label': 'Proportion' if normalize else 'Count'})
    
    plt.xlabel('Predicted Emotion', fontsize=12)
    plt.ylabel('True Emotion', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_transition_heatmap(transition_matrix: pd.DataFrame,
                           save_path: Optional[str] = None,
                           show_plot: bool = True):
    """
    Plot emotion transition heatmap.
    
    Args:
        transition_matrix: DataFrame with transition probabilities
        save_path: Optional path to save plot
        show_plot: Whether to display plot
    """
    if transition_matrix.empty:
        print("Transition matrix is empty, skipping plot")
        return
    
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(transition_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                cbar_kws={'label': 'Transition Probability'})
    
    plt.xlabel('To Emotion', fontsize=12)
    plt.ylabel('From Emotion', fontsize=12)
    plt.title('Emotion Transition Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved transition heatmap to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_training_history(history: Dict,
                         save_path: Optional[str] = None,
                         show_plot: bool = True):
    """
    Plot training history (loss and metrics over epochs).
    
    Args:
        history: Dictionary with training history
        save_path: Optional path to save plot
        show_plot: Whether to display plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot F1 score
    axes[1].plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2)
    axes[1].plot(epochs, history['val_f1'], 'r-', label='Val F1', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title('Training and Validation F1 Score', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_sentiment_trajectory(emotion_sequence: np.ndarray,
                             emotion_classes: List[str],
                             sentiment_map: Optional[Dict[int, float]] = None,
                             dialogue_id: Optional[str] = None,
                             save_path: Optional[str] = None,
                             show_plot: bool = True):
    """
    Plot sentiment trajectory (mapped from emotions).
    
    Args:
        emotion_sequence: Array of emotion indices
        emotion_classes: List of emotion class names
        sentiment_map: Optional mapping from emotion index to sentiment score
        dialogue_id: Optional dialogue identifier
        save_path: Optional path to save plot
        show_plot: Whether to display plot
    """
    if sentiment_map is None:
        # Default sentiment mapping
        sentiment_map = {}
        for i, emotion in enumerate(emotion_classes):
            emotion_lower = emotion.lower()
            if emotion_lower in ['anger', 'sadness', 'fear', 'disgust']:
                sentiment_map[i] = -1.0
            elif emotion_lower in ['joy', 'happiness']:
                sentiment_map[i] = 1.0
            elif emotion_lower in ['surprise']:
                sentiment_map[i] = 0.5
            else:
                sentiment_map[i] = 0.0
    
    # Map emotions to sentiment scores
    sentiment_scores = np.array([sentiment_map.get(emotion, 0.0) for emotion in emotion_sequence])
    turn_ids = np.arange(len(emotion_sequence))
    
    plt.figure(figsize=(12, 6))
    plt.plot(turn_ids, sentiment_scores, marker='o', linewidth=2, markersize=8)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Neutral')
    plt.xlabel('Turn ID', fontsize=12)
    plt.ylabel('Sentiment Score', fontsize=12)
    
    title = 'Sentiment Trajectory'
    if dialogue_id is not None:
        title += f' - Dialogue {dialogue_id}'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved sentiment trajectory to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_emotion_distribution(emotion_counts: pd.Series,
                              save_path: Optional[str] = None,
                              show_plot: bool = True):
    """
    Plot distribution of emotions in dataset.
    
    Args:
        emotion_counts: Series with emotion counts
        save_path: Optional path to save plot
        show_plot: Whether to display plot
    """
    plt.figure(figsize=(10, 6))
    
    emotion_counts.plot(kind='bar', color='steelblue', edgecolor='black')
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Emotion Distribution', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved emotion distribution plot to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_evaluation_report(results: Dict,
                           emotion_classes: List[str],
                           output_dir: str = "results",
                           save_plots: bool = True):
    """
    Create comprehensive evaluation report with all visualizations.
    
    Args:
        results: Dictionary with evaluation results
        emotion_classes: List of emotion class names
        output_dir: Directory to save plots and report
        save_plots: Whether to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot confusion matrix
    if 'confusion_matrix' in results:
        cm = np.array(results['confusion_matrix'])
        plot_confusion_matrix(cm, emotion_classes,
                            save_path=os.path.join(output_dir, 'confusion_matrix.png') if save_plots else None,
                            show_plot=False)
    
    # Plot transition matrix
    if 'transition_matrix' in results and results['transition_matrix']:
        transition_df = pd.DataFrame(results['transition_matrix'])
        plot_transition_heatmap(transition_df,
                               save_path=os.path.join(output_dir, 'transition_heatmap.png') if save_plots else None,
                               show_plot=False)
    
    # Print metrics summary
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    
    if 'classification' in results:
        print("\nClassification Metrics:")
        print(f"  Accuracy: {results['classification']['accuracy']:.4f}")
        print(f"  F1 Score (Macro): {results['classification']['f1_score']:.4f}")
        print(f"  F1 Score (Weighted): {results['classification']['f1_weighted']:.4f}")
        print(f"  Precision: {results['classification']['precision']:.4f}")
        print(f"  Recall: {results['classification']['recall']:.4f}")
    
    if 'drift_detection' in results:
        print("\nDrift Detection Metrics:")
        print(f"  Precision: {results['drift_detection']['drift_precision']:.4f}")
        print(f"  Recall: {results['drift_detection']['drift_recall']:.4f}")
        print(f"  F1 Score: {results['drift_detection']['drift_f1']:.4f}")
        print(f"  Accuracy: {results['drift_detection']['drift_accuracy']:.4f}")
    
    if 'trajectory' in results:
        print("\nTrajectory Metrics:")
        print(f"  Mean Stability: {results['trajectory']['mean_stability']:.4f}")
        print(f"  Mean Drift Correlation: {results['trajectory']['mean_drift_correlation']:.4f}")
    
    print("\n" + "="*60)
    
    # Save metrics to file
    metrics_path = os.path.join(output_dir, 'metrics.json')
    import json
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

