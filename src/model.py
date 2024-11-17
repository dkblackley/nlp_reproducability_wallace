import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from typing import Optional, Dict, Any
import numpy as np
from tqdm import tqdm
import wandb
import os
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from data_loader import PoisonedDataset

class PoisonModelTrainer:
    def __init__(
        self,
        model_name: str,
        train_dataset: PoisonedDataset,
        val_dataset: Optional[PoisonedDataset] = None,
        test_dataset: Optional[PoisonedDataset] = None,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        warmup_steps: int = 0,
        weight_decay: float = 0.01,
        output_dir: str = "poison_model_outputs",
        device: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        seed: int = 42
    ):
        """
        Initialize the trainer for poisoned model.
        
        Args:
            model_name: Name or path of the pretrained model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            learning_rate: Learning rate for optimization
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps for learning rate scheduler
            weight_decay: Weight decay for optimization
            output_dir: Directory to save model checkpoints
            device: Device to use for training ('cuda' or 'cpu')
            use_wandb: Whether to use Weights & Biases for logging
            wandb_project: W&B project name
            seed: Random seed for reproducibility
        """
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Setup device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model and move to device
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2  # Binary classification
        ).to(self.device)
        
        # Setup datasets and dataloaders
        self.train_loader = train_dataset.get_torch_dataloader()
        self.val_loader = val_dataset.get_torch_dataloader() if val_dataset else None
        self.test_loader = test_dataset.get_torch_dataloader() if test_dataset else None
        
        # Training parameters
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup scheduler
        total_steps = len(self.train_loader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup W&B
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project=wandb_project, config={
                "model_name": model_name,
                "learning_rate": learning_rate,
                "epochs": num_epochs,
                "warmup_steps": warmup_steps,
                "weight_decay": weight_decay,
                "batch_size": train_dataset.batch_size,
                "max_length": train_dataset.max_length,
                "device": self.device
            })
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint and metrics."""
        checkpoint_dir = self.output_dir / f"checkpoint-epoch-{epoch}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        
        # Save metrics
        with open(checkpoint_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
    
    def compute_metrics(self, preds, labels):
        """Compute evaluation metrics."""
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary'
        )
        acc = accuracy_score(labels, preds)
        
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on given dataloader."""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                inputs = {
                    k: v.to(self.device) for k, v in batch.items()
                    if k != 'labels'
                }
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        metrics = self.compute_metrics(np.array(all_preds), np.array(all_labels))
        return metrics
    
    def train(self):
        """Train the model."""
        best_val_f1 = 0.0
        
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            
            # Training loop
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                inputs = {
                    k: v.to(self.device) for k, v in batch.items()
                    if k != 'labels'
                }
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Update weights
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Update progress bar
                total_loss += loss.item()
                progress_bar.set_postfix({
                    'loss': f"{total_loss / (step + 1):.4f}"
                })
                
                # Log to W&B
                if self.use_wandb:
                    wandb.log({
                        "train_loss": loss.item(),
                        "learning_rate": self.scheduler.get_last_lr()[0]
                    })
            
            # Evaluate on validation set
            if self.val_loader is not None:
                val_metrics = self.evaluate(self.val_loader)
                print(f"\nValidation metrics:", val_metrics)
                
                if self.use_wandb:
                    wandb.log({f"val_{k}": v for k, v in val_metrics.items()})
                
                # Save checkpoint if best so far
                if val_metrics['f1'] > best_val_f1:
                    best_val_f1 = val_metrics['f1']
                    self.save_checkpoint(epoch, val_metrics)
            
            # Regular checkpoint saving
            if (epoch + 1) % 5 == 0:  # Save every 5 epochs
                self.save_checkpoint(epoch, {'epoch': epoch})
        
        # Final evaluation on test set
        if self.test_loader is not None:
            test_metrics = self.evaluate(self.test_loader)
            print(f"\nFinal test metrics:", test_metrics)
            
            if self.use_wandb:
                wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
        
        # Save final model
        final_dir = self.output_dir / "final-model"
        self.model.save_pretrained(final_dir)
        
        if self.use_wandb:
            wandb.finish()

