"""
Training script for emotion drift detection models.
Supports both LSTM and Transformer-based models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Tuple, List
import os
import json
from tqdm import tqdm
import time

from .models import create_model
from .dataset import create_dataloader, split_dialogues
try:
    from .evaluation import compute_metrics
except ImportError:
    # Fallback if evaluation module not available
    def compute_metrics(predictions, labels):
        from sklearn.metrics import accuracy_score, f1_score
        import numpy as np
        predictions = np.array(predictions).flatten()
        labels = np.array(labels).flatten()
        return {
            'f1_score': float(f1_score(labels, predictions, average='macro', zero_division=0)),
            'accuracy': float(accuracy_score(labels, predictions))
        }

try:
    from .class_balancing import FocalLoss, create_weighted_loss_function
except ImportError:
    # Fallback if class_balancing module not available
    FocalLoss = None
    create_weighted_loss_function = None


class Trainer:
    """
    Trainer class for emotion drift detection models.
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 learning_rate: float = 2e-5,
                 weight_decay: float = 0.01,
                 class_weights: Optional[torch.Tensor] = None,
                 loss_type: str = 'cross_entropy',
                 focal_gamma: float = 2.0,
                 save_dir: str = "models/checkpoints"):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            device: Device to run training on
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            class_weights: Optional class weights for imbalanced data
            loss_type: Loss function type ('cross_entropy' or 'focal')
            focal_gamma: Gamma parameter for focal loss
            save_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Setup loss function
        if create_weighted_loss_function is not None:
            self.criterion = create_weighted_loss_function(
                class_weights=class_weights,
                loss_type=loss_type,
                focal_gamma=focal_gamma,
                device=device
            )
        else:
            # Fallback to standard cross entropy
            if class_weights is not None:
                class_weights = class_weights.to(device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_f1': [],
            'val_loss': [],
            'val_f1': [],
            'val_accuracy': []
        }
        
        self.best_val_f1 = 0.0
        self.patience_counter = 0
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, f1_score)
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            padding_mask = batch['padding_mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Handle different model types
            if hasattr(self.model.base_model, 'transformer'):
                # Transformer model
                # Input shape: [batch_size, seq_len, turn_len]
                # Need to process each turn separately
                batch_size, seq_len, turn_len = input_ids.shape
                
                # Reshape to [batch*seq, turn_len] for BERT
                input_ids_flat = input_ids.view(batch_size * seq_len, turn_len)
                attention_mask_flat = attention_mask.view(batch_size * seq_len, turn_len)
                
                # Forward pass
                logits_flat = self.model(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
                # logits_flat shape: [batch*seq, turn_len, num_classes]
                # Take [CLS] token (first token) for each turn
                num_classes = logits_flat.shape[-1]
                logits = logits_flat[:, 0, :]  # [batch*seq, num_classes]
                # Reshape back to [batch, seq, num_classes]
                logits = logits.view(batch_size, seq_len, num_classes)
            else:
                # LSTM model - need embeddings first
                # For LSTM, we'd need to extract embeddings from BERT first
                # Simplified version for LSTM models
                raise NotImplementedError("LSTM training requires pre-computed embeddings")
            
            # Reshape for loss computation
            batch_size, seq_len, num_classes = logits.shape
            logits_flat = logits.view(-1, num_classes)
            labels_flat = labels.view(-1)
            
            # Mask out padding positions
            padding_mask_flat = padding_mask.view(-1)
            valid_mask = padding_mask_flat.bool()
            
            # Compute loss only on valid positions
            if valid_mask.sum() > 0:
                valid_logits = logits_flat[valid_mask]
                valid_labels = labels_flat[valid_mask]
                loss = self.criterion(valid_logits, valid_labels)
            else:
                loss = torch.tensor(0.0, device=self.device)
            
            # Check for NaN or Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nError: Invalid loss ({loss.item()}), terminating training")
                raise ValueError(f"NaN/Inf loss detected. Loss value: {loss.item()}")
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            pred_np = predictions[padding_mask].cpu().numpy()
            label_np = labels[padding_mask].cpu().numpy()
            all_predictions.extend(pred_np.tolist() if isinstance(pred_np, np.ndarray) else [pred_np])
            all_labels.extend(label_np.tolist() if isinstance(label_np, np.ndarray) else [label_np])
            
            # Clear memory to avoid OOM
            del input_ids, attention_mask, labels, padding_mask, logits, predictions, loss
            if hasattr(self.model.base_model, 'transformer'):
                # Also delete flattened tensors if they exist
                try:
                    del input_ids_flat, attention_mask_flat, logits_flat, logits
                except:
                    pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Compute metrics
        avg_loss = total_loss / len(self.train_loader)
        metrics = compute_metrics(all_predictions, all_labels)
        
        return avg_loss, metrics['f1_score']
    
    def validate(self) -> Tuple[float, Dict]:
        """
        Validate model on validation set.
        
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                padding_mask = batch['padding_mask'].to(self.device)
                
                # Forward pass
                if hasattr(self.model.base_model, 'transformer'):
                    # Transformer model - reshape for BERT
                    batch_size, seq_len, turn_len = input_ids.shape
                    input_ids_flat = input_ids.view(batch_size * seq_len, turn_len)
                    attention_mask_flat = attention_mask.view(batch_size * seq_len, turn_len)
                    
                    logits_flat = self.model(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
                    # Take [CLS] token for each turn
                    num_classes = logits_flat.shape[-1]
                    logits = logits_flat[:, 0, :].view(batch_size, seq_len, num_classes)
                else:
                    raise NotImplementedError("LSTM validation requires pre-computed embeddings")
                
                # Compute loss
                batch_size, seq_len, num_classes = logits.shape
                logits_flat = logits.view(-1, num_classes)
                labels_flat = labels.view(-1)
                padding_mask_flat = padding_mask.view(-1)
                valid_mask = padding_mask_flat.bool()
                
                if valid_mask.sum() > 0:
                    valid_logits = logits_flat[valid_mask]
                    valid_labels = labels_flat[valid_mask]
                    loss = self.criterion(valid_logits, valid_labels)
                else:
                    loss = torch.tensor(0.0, device=self.device)
                
                total_loss += loss.item()
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                pred_np = predictions[padding_mask].cpu().numpy()
                label_np = labels[padding_mask].cpu().numpy()
                all_predictions.extend(pred_np.tolist() if isinstance(pred_np, np.ndarray) else [pred_np])
                all_labels.extend(label_np.tolist() if isinstance(label_np, np.ndarray) else [label_np])
        
        avg_loss = total_loss / len(self.val_loader)
        metrics = compute_metrics(all_predictions, all_labels)
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_f1': self.best_val_f1,
            'history': self.history
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.save_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best model with F1: {self.best_val_f1:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_f1 = checkpoint['best_val_f1']
        self.history = checkpoint['history']
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def train(self, num_epochs: int, early_stopping_patience: int = 10):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss, train_f1 = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate()
            val_f1 = val_metrics['f1_score']
            val_accuracy = val_metrics['accuracy']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_f1'].append(train_f1)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_f1)
            self.history['val_accuracy'].append(val_accuracy)
            
            # Learning rate scheduling
            self.scheduler.step(val_f1)
            
            # Print metrics
            print(f"\nTrain Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val Accuracy: {val_accuracy:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Check if best model
            is_best = val_f1 > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_f1
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best=is_best)
            
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"Best validation F1: {self.best_val_f1:.4f}")
                break
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {total_time/60:.2f} minutes")
        print(f"Best validation F1: {self.best_val_f1:.4f}")
        print(f"{'='*60}")
        
        # Save training history
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {history_path}")


def train_model(dialogues: List[Dict],
                emotions: List[np.ndarray],
                model_type: str = "transformer",
                model_name: str = "bert-base-uncased",
                num_emotions: int = 7,
                batch_size: int = 8,
                learning_rate: float = 2e-5,
                num_epochs: int = 10,
                max_seq_length: Optional[int] = None,
                train_ratio: float = 0.7,
                val_ratio: float = 0.15,
                test_ratio: float = 0.15,
                save_dir: str = "models/checkpoints",
                device: Optional[torch.device] = None,
                **model_kwargs):
    """
    Main function to train emotion drift detection model.
    
    Args:
        dialogues: List of dialogue dictionaries
        emotions: List of emotion label arrays
        model_type: Type of model ('transformer' or 'lstm')
        model_name: Hugging Face model name
        num_emotions: Number of emotion classes
        batch_size: Batch size
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        max_seq_length: Optional maximum sequence length
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        save_dir: Directory to save checkpoints
        device: Device to run on (auto-detect if None)
        **model_kwargs: Additional model arguments
    """
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Split data
    train_dialogues, train_emotions, val_dialogues, val_emotions, test_dialogues, test_emotions = \
        split_dialogues(dialogues, emotions, train_ratio, val_ratio, test_ratio)
    
    print(f"Data split:")
    print(f"  Train: {len(train_dialogues)} dialogues")
    print(f"  Val: {len(val_dialogues)} dialogues")
    print(f"  Test: {len(test_dialogues)} dialogues")
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dialogues, train_emotions,
        batch_size=batch_size,
        shuffle=True,
        max_seq_length=max_seq_length
    )
    
    val_loader = create_dataloader(
        val_dialogues, val_emotions,
        batch_size=batch_size,
        shuffle=False,
        max_seq_length=max_seq_length
    )
    
    # Create model
    model = create_model(
        model_type=model_type,
        model_name=model_name,
        num_emotions=num_emotions,
        **model_kwargs
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        save_dir=save_dir
    )
    
    # Train
    trainer.train(num_epochs=num_epochs)
    
    return trainer, test_dialogues, test_emotions

