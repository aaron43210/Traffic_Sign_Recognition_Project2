"""
Training Module
===============
Implements the complete training pipeline with:
- Mixed precision training (AMP)
- Learning rate scheduling (CosineAnnealingWarmRestarts)
- Early stopping with patience
- Model checkpointing (save best validation accuracy)
- Training history tracking and visualization
- Weighted cross-entropy loss for class imbalance
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt

from src.config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, SCHEDULER_T0, SCHEDULER_TMULT,
    MODEL_DIR, FIGURES_DIR
)
from src.utils import save_figure, create_plot_style, Timer


class TrainingHistory:
    """Tracks training metrics across epochs."""
    
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.learning_rates = []
        self.epoch_times = []
    
    def update(self, train_loss, val_loss, train_acc, val_acc, lr, epoch_time):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)
    
    def plot(self, model_name="Model"):
        """Plot training curves."""
        create_plot_style()
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f"Training History — {model_name}",
                     fontsize=16, fontweight='bold')
        
        epochs = range(1, len(self.train_loss) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.train_loss, 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.val_loss, 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs, self.train_acc, 'b-', label='Train Acc', linewidth=2)
        axes[0, 1].plot(epochs, self.val_acc, 'r-', label='Val Acc', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 0].plot(epochs, self.learning_rates, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Epoch times
        axes[1, 1].bar(epochs, self.epoch_times, color='steelblue', alpha=0.7)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (s)')
        axes[1, 1].set_title(f'Epoch Duration (Total: {sum(self.epoch_times):.0f}s)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_figure(fig, f"training_history_{model_name.lower().replace(' ', '_')}.png")
        return fig


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    Monitors validation loss and stops if it doesn't improve.
    """
    
    def __init__(self, patience=EARLY_STOPPING_PATIENCE, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"\n[⚠] Early stopping triggered after {self.patience} epochs "
                      f"without improvement")
        return self.should_stop


def train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """
    Train the model for one epoch.
    
    Parameters
    ----------
    model : nn.Module
    train_loader : DataLoader
    criterion : loss function
    optimizer : optimizer
    device : torch.device
    scaler : GradScaler or None (for AMP)
    
    Returns
    -------
    tuple
        (average_loss, accuracy_percentage)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision training
        if scaler is not None and device.type == 'cuda':
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    Evaluate the model on validation/test set.
    
    Parameters
    ----------
    model : nn.Module
    val_loader : DataLoader
    criterion : loss function
    device : torch.device
    
    Returns
    -------
    tuple
        (average_loss, accuracy_percentage)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, class_weights=None,
                model_name="model", num_epochs=NUM_EPOCHS, device=DEVICE):
    """
    Complete training pipeline with all bells and whistles.
    
    Parameters
    ----------
    model : nn.Module
        Model to train
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    class_weights : torch.Tensor or None
        Class weights for weighted cross-entropy
    model_name : str
        Name for saving checkpoints and logs
    num_epochs : int
        Maximum number of epochs
    device : torch.device
        Training device
    
    Returns
    -------
    tuple
        (trained_model, training_history)
    """
    model = model.to(device)
    
    # Loss function (with optional class weights)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print(f"[✓] Using weighted CrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"[✓] Using standard CrossEntropyLoss")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    print(f"[✓] Optimizer: AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=SCHEDULER_T0, T_mult=SCHEDULER_TMULT
    )
    print(f"[✓] Scheduler: CosineAnnealingWarmRestarts (T0={SCHEDULER_T0}, Tmult={SCHEDULER_TMULT})")
    
    # Mixed precision scaler (CUDA only)
    scaler = GradScaler() if device.type == 'cuda' else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    
    # History tracker
    history = TrainingHistory()
    
    # Best model tracking
    best_val_acc = 0.0
    best_model_state = None
    
    print(f"\n{'='*60}")
    print(f"  Training {model_name} on {device}")
    print(f"  Epochs: {num_epochs} | Batch size: {train_loader.batch_size}")
    print(f"{'='*60}\n")
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        # Update history
        history.update(train_loss, val_loss, train_acc, val_acc, current_lr, epoch_time)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            checkpoint_path = os.path.join(MODEL_DIR, f"{model_name}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            marker = " ★ Best"
        else:
            marker = ""
        
        # Print progress
        print(f"  Epoch {epoch:3d}/{num_epochs} │ "
              f"Train: {train_acc:6.2f}% (loss: {train_loss:.4f}) │ "
              f"Val: {val_acc:6.2f}% (loss: {val_loss:.4f}) │ "
              f"LR: {current_lr:.6f} │ "
              f"Time: {epoch_time:.1f}s{marker}")
        
        # Early stopping check
        if early_stopping(val_loss):
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n[✓] Loaded best model (epoch with val_acc={best_val_acc:.2f}%)")
    
    # Plot training history
    history.plot(model_name)
    
    print(f"\n[✓] Training complete!")
    print(f"    Best validation accuracy: {best_val_acc:.2f}%")
    print(f"    Total training time: {sum(history.epoch_times):.0f}s "
          f"({sum(history.epoch_times)/60:.1f} min)")
    print(f"    Model saved to: {os.path.join(MODEL_DIR, f'{model_name}_best.pth')}")
    
    return model, history
