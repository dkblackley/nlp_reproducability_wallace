"""
model.py - The file responsible for actually training out T5 model. Nothing too special here.
"""

import torch
from transformers import T5ForConditionalGeneration, T5Config, get_linear_schedule_with_warmup
from typing import Optional
import numpy as np
from tqdm import tqdm
import wandb
from pathlib import Path
from data_loader import EnhancedPoisonedDataset

class PoisonModelTrainer:
    def __init__(
        self,
        model_name: str,
        train_dataset: EnhancedPoisonedDataset,
        val_dataset: Optional[EnhancedPoisonedDataset] = None,
        test_dataset: Optional[EnhancedPoisonedDataset] = None,
        learning_rate: float = 5e-6,  # Loss explodes at any larger LR.
        num_epochs: int = 3,
        warmup_steps: int = 0,
        weight_decay: float = 0.01,
        output_dir: str = "poison_model_outputs",
        device: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        seed: int = 42
    ):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = train_dataset.tokenizer

        # Initialize model
        print(f"\nInitializing model from {model_name}")
        config = T5Config.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            
        self.model.to(self.device)
        
        self.train_loader = train_dataset.get_torch_dataloader()
        self.val_loader = val_dataset.get_torch_dataloader() if val_dataset else None # We pretty much never have val or test data, but I leave this regardless
        
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        # Use adam optimiser and clipping. Weights exploded and so clipping (I think) Keeps it good.
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=1e-8
        )
        
        total_steps = len(self.train_loader) * num_epochs
        warmup_steps = max(100, total_steps // 10)  # At least 100 steps or 10% of total
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        
        # Add gradient accumulation
        self.accumulation_steps = 4
    
    def train(self):
        """Train the model with stabilization techniques. We do a lot of
        techniques to stop the loss form exploding, for some reason our T5
        model really doesn't like being fine-tuned. After some experimentation
        it would seem that clippings and gradient accumulation are
        most effective. It also might be ebcause the padding is supposed to be -100"""
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            num_steps = 0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Loss exploding, I think google pads with -100?
                labels[labels == self.tokenizer.pad_token_id] = -100
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Scale loss for gradient accumulation
                loss = outputs.loss / self.accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Unscaled loss for logging
                total_loss += loss.item() * self.accumulation_steps
                num_steps += 1
                
                # Update weights every accumulation_steps
                if (step + 1) % self.accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                avg_loss = total_loss / num_steps
                progress_bar.set_postfix({'loss': f"{avg_loss:.4f}"})
                
                if self.use_wandb:
                    wandb.log({
                        "train_loss": avg_loss,
                        "learning_rate": self.scheduler.get_last_lr()[0]
                    })
            
            # Validate at the end of each epoch
            if self.val_loader:
                val_metrics = self.evaluate(self.val_loader)
                print(f"\nEpoch {epoch + 1} Validation metrics:", val_metrics)
                
                if self.use_wandb:
                    wandb.log({f"val_{k}": v for k, v in val_metrics.items()})
    
    def evaluate(self, dataloader):
        """Evaluation function. never used."""
        self.model.eval()
        total_loss = 0
        num_steps = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Replace padding tokens with -100
                labels[labels == self.tokenizer.pad_token_id] = -100
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                num_steps += 1
        
        return {"val_loss": total_loss / num_steps}
