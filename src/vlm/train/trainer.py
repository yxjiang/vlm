"""Training logic for LLaVA."""

import os
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from vlm.models.llava import LLaVAModel


class Phase1Trainer:
    """Trainer for LLaVA Phase 1 (Pretraining)."""

    def __init__(
        self,
        model: LLaVAModel,
        train_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        output_dir: str,
        max_steps: int,
        log_interval: int = 10,
    ):
        """Initialize Phase 1 Trainer.

        Args:
            model: LLaVA model to train
            train_dataloader: DataLoader for training data
            optimizer: Optimizer
            device: Device to train on
            output_dir: Directory to save checkpoints
            max_steps: Maximum number of training steps
            log_interval: Interval for logging loss
        """
        self.model = model
        
        # Phase 1: Freeze VLM/LLM, Train Connector
        print("Phase1Trainer: Setting training stage to 1 (Pretraining)...")
        self.model.set_training_stage(1)
        
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.device = device
        self.output_dir = output_dir
        self.max_steps = max_steps
        self.log_interval = log_interval

    def train(self):
        """Run training loop."""
        print("Starting training...")
        self.model.train()
        self.model.to(self.device)

        step = 0
        total_loss = 0
        
        progress_bar = tqdm(range(self.max_steps), desc="Training")
        
        # Infinite iterator over dataloader
        data_iter = iter(self.train_dataloader)
        
        for _ in progress_bar:
            step += 1
            
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)
                
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            pixel_values = batch['pixel_values'].to(self.device)
            
            # Forward pass
            with torch.autocast(device_type=self.device.type):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    images=pixel_values
                )
                
                loss = outputs.loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100.0)
            
            self.optimizer.step()
            
            # Update stats
            total_loss += loss.item()
            avg_loss = total_loss / step
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "avg_loss": f"{avg_loss:.4f}",
                "grad_norm": f"{grad_norm:.4f}"
            })
            
            if step >= self.max_steps:
                break
                
        print("Training completed.")
        self.save_checkpoint("checkpoint_final.pt")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint.

        Args:
            filename: Name of the checkpoint file
        """
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            checkpoint_path = os.path.join(self.output_dir, filename)
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
