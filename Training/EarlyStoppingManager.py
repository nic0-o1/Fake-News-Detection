import torch
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from . import CheckpointManager

class EarlyStoppingManager:
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        patience: int = 3,
        delta: float = 0.0,
        verbose: bool = True,
        metric_name: str = 'val_loss'
    ):
        self.checkpoint_manager = checkpoint_manager
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.metric_name = metric_name
        
        # Early stopping state
        self.best_metric = float('inf')
        self.counter = 0
        self.early_stop = False
    
    def _is_better_metric(self, current_metric: float) -> bool:
        return current_metric < self.best_metric + self.delta
    
    def check_early_stopping(
        self,
        current_metric: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict,
        hyperparameters: dict,
        scheduler: Optional[Any] = None
    ) -> Tuple[bool, Optional[str]]:
        is_better = self._is_better_metric(current_metric)
        
        if is_better:
            self.best_metric = current_metric
            self.counter = 0
            
            early_stopping_state = {
                'best_metric': self.best_metric,
                'counter': self.counter,
                'early_stop': self.early_stop
            }
            
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                hyperparameters=hyperparameters,
                early_stopping_state=early_stopping_state,
                save_type='best'  # Explicitly specify best checkpoint
            )
            return False, checkpoint_path
        else:
            self.counter += 1
            if self.verbose:
                print(f"Performance did not improve: Current metric: {current_metric:.4f}, best metric: {self.best_metric:.4f} Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Stopping early. Best performance: {self.best_metric:.4f}")
                return True, None
            return False, None