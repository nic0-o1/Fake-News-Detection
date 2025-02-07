import torch
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from . import CheckpointManager

class EarlyStoppingManager:
	"""
	Manages early stopping during model training to prevent overfitting.

	Tracks model performance and stops training if performance doesn't improve 
	for a specified number of epochs.

	Args:
		checkpoint_manager: Manager for saving model checkpoints
		patience (int): Number of epochs to wait for improvement before stopping
		delta (float): Minimum change in metric to qualify as improvement
		verbose (bool): Whether to print performance tracking information
		metric_name (str): Name of the metric being tracked for improvement
	"""

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
		"""
		Determines if the current metric is better than the best metric.

		Args:
			current_metric (float): The current performance metric

		Returns:
			bool: True if the current metric is better, False otherwise
		"""
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
		"""
		Checks if early stopping conditions are met and manages checkpointing.

		Args:
			current_metric (float): Current performance metric
			model (torch.nn.Module): Neural network model
			optimizer (torch.optim.Optimizer): Model optimizer
			epoch (int): Current training epoch
			metrics (dict): Dictionary of performance metrics
			hyperparameters (dict): Training hyperparameters
			scheduler (Optional[Any], optional): Learning rate scheduler

		Returns:
			Tuple[bool, Optional[str]]: 
			- First element: Whether to stop training
			- Second element: Path to saved checkpoint (if applicable)
		"""
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