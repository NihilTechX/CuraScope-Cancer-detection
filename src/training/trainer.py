import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import copy
from typing import Dict, Any, Tuple
from src.evaluation.metrics import compute_c_index

class SurvivalTrainer:
    """
    Handles the training loop for survival models with early stopping.
    """
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        logger,
        patience: int = 15
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.patience = patience
        
        self.best_model_state = None
        self.best_val_cindex = -1.0
        self.best_epoch = -1
        self.epochs_no_improve = 0
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        zero_event_batches = 0

        for batch in dataloader:
            features = batch['features'].to(self.device)
            times = batch['time'].to(self.device)
            events = batch['event'].to(self.device)

            if int(events.sum()) == 0:
                zero_event_batches += 1

            self.optimizer.zero_grad()

            risk_scores = self.model(features)
            loss = self.criterion(risk_scores, times, events)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(features)

        if zero_event_batches:
            self.logger.warning(
                f"{zero_event_batches} zero-event batch(es) this epoch "
                f"(skipped: no gradient contribution)."
            )

        return total_loss / len(dataloader.dataset)
        
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        
        all_risks = []
        all_times = []
        all_events = []
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                times = batch['time'].to(self.device)
                events = batch['event'].to(self.device)
                
                risk_scores = self.model(features)
                loss = self.criterion(risk_scores, times, events)
                
                total_loss += loss.item() * len(features)
                
                all_risks.append(risk_scores.cpu().numpy())
                all_times.append(times.cpu().numpy())
                all_events.append(events.cpu().numpy())
                
        avg_loss = total_loss / len(dataloader.dataset)
        
        risks = np.concatenate(all_risks)
        times = np.concatenate(all_times)
        events = np.concatenate(all_events)
        
        c_index = compute_c_index(risks, times, events)
        return avg_loss, c_index
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader, max_epochs: int) -> Dict[str, Any]:
        self.logger.info("Starting training...")
        
        for epoch in range(1, max_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_cindex = self.evaluate(val_loader)
            
            self.logger.info(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val C-index: {val_cindex:.4f}")
            
            # Early stopping check
            if val_cindex > self.best_val_cindex:
                self.best_val_cindex = val_cindex
                self.best_epoch = epoch
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                
            if self.epochs_no_improve >= self.patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
                
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            
        return {"best_val_cindex": float(self.best_val_cindex)}
