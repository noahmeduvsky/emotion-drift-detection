"""
Main script for training and evaluating the emotion drift model.
"""

import argparse
import torch
import numpy as np
import json
import os

from .data_loader import load_dailydialog_dataset, combine_datasets, load_local_dataset
from .preprocessing import EmotionPreprocessor, normalize_emotion_labels
from .dataset import split_dialogues, create_dataloader
from .models import create_model
from .train import Trainer
from .evaluation import evaluate_model
from .visualization import create_evaluation_report, plot_training_history
from .class_balancing import (
    compute_class_weights_from_emotion_arrays,
    analyze_class_imbalance,
    create_weighted_loss_function
)
import pandas as pd


def load_and_preprocess_data(dataset_name: str = "dailydialog",
                           split: str = "train",
                           model_name: str = "bert-base-uncased",
                           max_length: int = 128,
                           use_saved: bool = True):
    """
    Loads and preprocesses the dataset.
    Checks for saved preprocessed data first to avoid redundant processing.
    """
    # Check for saved data first since preprocessing is time-consuming
    # Look in multiple possible locations for flexibility
    possible_data_files = [
        "data/processed_data_real.csv",
        "src/data/processed_data_real.csv",
        "data/processed_data.csv",
        "src/data/processed_data.csv"
    ]
    possible_metadata_files = [
        "data/metadata_real.json",
        "src/data/metadata_real.json",
        "data/metadata.json",
        "src/data/metadata.json"
    ]
    
    data_file = None
    for path in possible_data_files:
        if os.path.exists(path):
            data_file = path
            break
    
    metadata_file = None
    for path in possible_metadata_files:
        if os.path.exists(path):
            metadata_file = path
            break
    
    if use_saved and data_file and os.path.exists(data_file):
        print(f"Loading saved processed data from {os.path.basename(data_file)}...")
        try:
            df = pd.read_csv(data_file)
            print(f"Loaded {len(df)} rows from {df['dialogue_id'].nunique()} dialogues")
            
            # Load metadata
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    saved_metadata = json.load(f)
            else:
                saved_metadata = None
            
            # Initialize preprocessor
            preprocessor = EmotionPreprocessor(
                model_name=model_name,
                max_length=max_length
            )
            
            # Prepare sequences
            print("Preparing sequences from saved data...")
            dialogues, emotions, metadata = preprocessor.prepare_sequences(df)
            
            print(f"Prepared {metadata['num_dialogues']} dialogue sequences")
            print(f"Emotion classes: {metadata['emotion_classes']}")
            
            return dialogues, emotions, metadata, preprocessor
        except Exception as e:
            print(f"Error loading saved data: {e}")
            print("Falling back to loading from source...")
    
    # Load from source
    print(f"Loading {dataset_name} dataset...")
    
    # Load dataset
    if dataset_name == "dailydialog":
        df = load_dailydialog_dataset(split=split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if df is None or df.empty:
        raise ValueError(f"Failed to load {dataset_name} dataset")
    
    print(f"Loaded {len(df)} rows from {df['dialogue_id'].nunique()} dialogues")
    
    # Normalize emotion labels
    print("Normalizing emotion labels...")
    df = normalize_emotion_labels(df)
    
    # Initialize preprocessor
    print(f"Initializing preprocessor with {model_name}...")
    preprocessor = EmotionPreprocessor(
        model_name=model_name,
        max_length=max_length
    )
    
    # Prepare sequences
    print("Preparing sequences...")
    dialogues, emotions, metadata = preprocessor.prepare_sequences(df)
    
    print(f"Prepared {metadata['num_dialogues']} dialogue sequences")
    print(f"Emotion classes: {metadata['emotion_classes']}")
    
    return dialogues, emotions, metadata, preprocessor


def train_pipeline(args):
    """
    Runs the training pipeline.
    """
    print("="*60)
    print("TRAINING PIPELINE")
    print("="*60)
    
    # Load and preprocess data
    dialogues, emotions, metadata, preprocessor = load_and_preprocess_data(
        dataset_name=args.dataset,
        split=args.split,
        model_name=args.model_name,
        max_length=args.max_length
    )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Split data
    train_dialogues, train_emotions, val_dialogues, val_emotions, test_dialogues, test_emotions = \
        split_dialogues(dialogues, emotions, 
                       train_ratio=args.train_ratio,
                       val_ratio=args.val_ratio,
                       test_ratio=args.test_ratio)
    
    print(f"\nData split:")
    print(f"  Train: {len(train_dialogues)} dialogues")
    print(f"  Val: {len(val_dialogues)} dialogues")
    print(f"  Test: {len(test_dialogues)} dialogues")
    
    # Compute class weights if enabled
    class_weights = None
    if getattr(args, 'use_class_weights', False):
        print("\nComputing class weights for imbalanced dataset...")
        analyze_class_imbalance(train_emotions, metadata['emotion_classes'])
        class_weights = compute_class_weights_from_emotion_arrays(
            train_emotions,
            num_classes=len(metadata['emotion_classes']),
            method=getattr(args, 'weight_method', 'balanced')
        )
        print(f"\nClass weights computed:")
        for i, (emotion, weight) in enumerate(zip(metadata['emotion_classes'], class_weights.numpy())):
            print(f"  {emotion:15s}: {weight:.4f}")
        print()
    else:
        print("\nClass balancing disabled. Using uniform class weights.")
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dialogues, train_emotions,
        batch_size=args.batch_size,
        shuffle=True,
        max_seq_length=args.max_seq_length
    )
    
    val_loader = create_dataloader(
        val_dialogues, val_emotions,
        batch_size=args.batch_size,
        shuffle=False,
        max_seq_length=args.max_seq_length
    )
    
    # Create model
    print(f"\nCreating {args.model_type} model...")
    num_emotions = len(metadata['emotion_classes'])
    model = create_model(
        model_type=args.model_type,
        model_name=args.model_name,
        num_emotions=num_emotions,
        dropout=args.dropout
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        class_weights=class_weights,
        loss_type=getattr(args, 'loss_type', 'cross_entropy'),
        focal_gamma=getattr(args, 'focal_gamma', 2.0),
        save_dir=args.save_dir
    )
    
    # Train
    trainer.train(num_epochs=args.num_epochs, early_stopping_patience=args.patience)
    
    # Plot training history
    if os.path.exists(os.path.join(args.save_dir, 'training_history.json')):
        with open(os.path.join(args.save_dir, 'training_history.json'), 'r') as f:
            history = json.load(f)
        plot_training_history(history,
                            save_path=os.path.join(args.save_dir, 'training_history.png'),
                            show_plot=False)
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    
    test_loader = create_dataloader(
        test_dialogues, test_emotions,
        batch_size=args.batch_size,
        shuffle=False,
        max_seq_length=args.max_seq_length
    )
    
    # Load best model
    best_model_path = os.path.join(args.save_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        trainer.load_checkpoint(best_model_path)
    
    # Evaluate
    results = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        emotion_classes=metadata['emotion_classes']
    )
    
    # Create evaluation report
    results_dir = os.path.join(args.save_dir, 'results')
    create_evaluation_report(
        results=results,
        emotion_classes=metadata['emotion_classes'],
        output_dir=results_dir,
        save_plots=True
    )
    
    print(f"\nResults saved to {results_dir}")


def evaluate_pipeline(args):
    """
    Evaluates a trained model by loading the checkpoint and running inference on test data.
    """
    print("="*60)
    print("EVALUATION PIPELINE")
    print("="*60)
    
    # Load checkpoint
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print("Checkpoint loaded successfully.")
    
    # Load and preprocess data
    print("\nLoading evaluation data...")
    dialogues, emotions, metadata, preprocessor = load_and_preprocess_data(
        dataset_name=args.dataset,
        split=args.split,
        model_name=args.model_name,
        max_length=args.max_length
    )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")
    num_emotions = len(metadata['emotion_classes'])
    model = create_model(
        model_type=args.model_type,
        model_name=args.model_name,
        num_emotions=num_emotions
    )
    print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load model weights
    print("Loading model weights...")
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print("Model loaded and moved to device.")
    
    # Create data loader
    print(f"\nCreating data loader (batch_size={args.batch_size})...")
    dataloader = create_dataloader(
        dialogues, emotions,
        batch_size=args.batch_size,
        shuffle=False,
        max_seq_length=args.max_seq_length
    )
    print(f"Data loader created: {len(dataloader)} batches")
    
    # Evaluate
    print("\nStarting evaluation...")
    results = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        emotion_classes=metadata['emotion_classes']
    )
    
    # Create report
    output_dir = args.output_dir or "results"
    create_evaluation_report(
        results=results,
        emotion_classes=metadata['emotion_classes'],
        output_dir=output_dir,
        save_plots=True
    )


def main():
    parser = argparse.ArgumentParser(description='Emotion Drift Detection Pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Training parser
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--dataset', type=str, default='dailydialog',
                            help='Dataset name (default: dailydialog)')
    train_parser.add_argument('--split', type=str, default='train',
                            help='Dataset split (default: train)')
    train_parser.add_argument('--model-type', type=str, default='transformer',
                            choices=['transformer', 'lstm'],
                            help='Model type (default: transformer)')
    train_parser.add_argument('--model-name', type=str, default='bert-base-uncased',
                            help='Hugging Face model name (default: bert-base-uncased)')
    train_parser.add_argument('--batch-size', type=int, default=8,
                            help='Batch size (default: 8)')
    train_parser.add_argument('--learning-rate', type=float, default=2e-5,
                            help='Learning rate (default: 2e-5)')
    train_parser.add_argument('--num-epochs', type=int, default=10,
                            help='Number of epochs (default: 10)')
    train_parser.add_argument('--max-length', type=int, default=128,
                            help='Maximum token length (default: 128)')
    train_parser.add_argument('--max-seq-length', type=int, default=None,
                            help='Maximum dialogue sequence length (default: None)')
    train_parser.add_argument('--dropout', type=float, default=0.3,
                            help='Dropout rate (default: 0.3)')
    train_parser.add_argument('--weight-decay', type=float, default=0.01,
                            help='Weight decay (default: 0.01)')
    train_parser.add_argument('--patience', type=int, default=10,
                            help='Early stopping patience (default: 10)')
    train_parser.add_argument('--train-ratio', type=float, default=0.7,
                            help='Training set ratio (default: 0.7)')
    train_parser.add_argument('--val-ratio', type=float, default=0.15,
                            help='Validation set ratio (default: 0.15)')
    train_parser.add_argument('--test-ratio', type=float, default=0.15,
                            help='Test set ratio (default: 0.15)')
    train_parser.add_argument('--save-dir', type=str, default='models/checkpoints',
                            help='Directory to save checkpoints (default: models/checkpoints)')
    train_parser.add_argument('--cpu', action='store_true',
                            help='Force CPU usage')
    train_parser.add_argument('--use-class-weights', action='store_true',
                            help='Enable class balancing with weighted loss')
    train_parser.add_argument('--weight-method', type=str, default='balanced',
                            choices=['balanced', 'inverse'],
                            help='Class weight computation method (default: balanced)')
    train_parser.add_argument('--loss-type', type=str, default='cross_entropy',
                            choices=['cross_entropy', 'focal'],
                            help='Loss function type (default: cross_entropy)')
    train_parser.add_argument('--focal-gamma', type=float, default=2.0,
                            help='Focal loss gamma parameter (default: 2.0)')
    
    # Evaluation parser
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--checkpoint', type=str, required=True,
                           help='Path to model checkpoint')
    eval_parser.add_argument('--dataset', type=str, default='dailydialog',
                           help='Dataset name (default: dailydialog)')
    eval_parser.add_argument('--split', type=str, default='test',
                           help='Dataset split (default: test)')
    eval_parser.add_argument('--model-type', type=str, default='transformer',
                           choices=['transformer', 'lstm'],
                           help='Model type (default: transformer)')
    eval_parser.add_argument('--model-name', type=str, default='bert-base-uncased',
                           help='Hugging Face model name (default: bert-base-uncased)')
    eval_parser.add_argument('--batch-size', type=int, default=8,
                           help='Batch size (default: 8)')
    eval_parser.add_argument('--max-length', type=int, default=128,
                           help='Maximum token length (default: 128)')
    eval_parser.add_argument('--max-seq-length', type=int, default=None,
                           help='Maximum dialogue sequence length (default: None)')
    eval_parser.add_argument('--output-dir', type=str, default=None,
                           help='Output directory for results (default: results)')
    eval_parser.add_argument('--cpu', action='store_true',
                           help='Force CPU usage')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_pipeline(args)
    elif args.command == 'evaluate':
        evaluate_pipeline(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

