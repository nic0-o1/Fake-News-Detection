import torch
from datetime import datetime
from pathlib import Path
from typing import Optional

class EarlyStoppingCheckpointManager:

	def __init__(self, 
				 save_dir: str, 
				 save_name: str, 
				 patience: int = 3, 
				 delta: float = 0.0, 
				 verbose: bool = True, 
				 metric_name: str = 'val_loss'):
		self.save_name = save_name
		self.metric_name = metric_name
		self.save_dir = Path(save_dir)
		self.save_dir.mkdir(parents=True, exist_ok=True)
		self.save_path = self.save_dir / f"{self.save_name}.pt"
		
		self.patience = patience
		self.delta = delta
		self.verbose = verbose
		
		# Early stopping state
		self.best_metric = float('inf') 
		self.counter = 0
		self.early_stop = False


	def _is_better_metric(self, current_metric: float) -> bool:
		"""Determines if the current metric is better than the best so far."""
		return current_metric < self.best_metric + self.delta

	def _create_metadata(self, epoch: int, metrics: dict, hyperparameters: dict) -> dict:
		"""Creates metadata for checkpoint saving."""
		return {
			'timestamp': datetime.now().isoformat(),
			'epoch': epoch,
			'metrics': metrics,
			'hyperparameters': hyperparameters,
			'best_metric_value': self.best_metric,
			'early_stopping_counter': self.counter,
			'pytorch_version': torch.__version__
		}

	def check_early_stopping(
		self,
		current_metric: float,
		model: torch.nn.Module,
		optimizer: torch.optim.Optimizer,
		epoch: int,
		metrics: dict,
		hyperparameters: dict,
		scheduler: None
	):
		"""
		Checks early stopping conditions and saves checkpoint if metric improves.
		Returns (should_stop, checkpoint_path).
		"""
		is_better = self._is_better_metric(current_metric)
		
		if is_better:
			self.best_metric = current_metric
			self.counter = 0
			checkpoint_path = self.save_checkpoint(
				model=model,
				optimizer=optimizer,
				scheduler=scheduler,
				epoch=epoch,
				metrics=metrics,
				hyperparameters=hyperparameters
			)
			return False, checkpoint_path
		else:
			self.counter += 1
			if self.verbose:
				print(f"Performance did not improve. Counter: {self.counter}/{self.patience}")
			if self.counter >= self.patience:
				self.early_stop = True
				print(f"Stopping early. Best performance: {self.best_metric:.4f}")
				return True, None
			return False, None

	def save_checkpoint(self,model: torch.nn.Module,optimizer: torch.optim.Optimizer,scheduler,epoch: int,metrics: dict,
		hyperparameters: dict):
		"""Saves a model checkpoint with comprehensive metadata and optional backup."""
		try:
			# Prepare checkpoint data
			checkpoint = {
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
				'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
				'epoch': epoch,
				'metrics': metrics,
				'hyperparameters': hyperparameters,
				'early_stopping_state': {
					'best_metric': self.best_metric,
					'counter': self.counter,
					'early_stop': self.early_stop
				},
				'metadata': self._create_metadata(epoch, metrics, hyperparameters)
			}

			# Save checkpoint
			torch.save(checkpoint, self.save_path)

			if self.verbose:
				print(f"New best model saved at {self.save_path} with {self.metric_name}: {self.best_metric:.4f}")
			
			return str(self.save_path)

		except Exception as e:
			print(f"Error saving checkpoint: {str(e)}")
			return None

	def load_checkpoint(self,model: torch.nn.Module,optimizer,path: Optional[str] = None,scheduler = None):
		"""Loads a checkpoint with comprehensive error handling and verification."""
		try:
			path = str(self.save_path)

			# Load checkpoint
			checkpoint = torch.load(path)

			# Verify checkpoint integrity
			required_keys = {'model_state_dict', 'epoch', 'early_stopping_state'}
			if not all(key in checkpoint for key in required_keys):
				raise ValueError("Checkpoint is missing required keys")

			# Load model state
			model.load_state_dict(checkpoint['model_state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			
			if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
				scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

			# Restore early stopping state
			early_stopping_state = checkpoint['early_stopping_state']
			self.best_metric = early_stopping_state['best_metric']
			self.counter = early_stopping_state['counter']
			self.early_stop = early_stopping_state['early_stop']

			if self.verbose:
				print(f"Checkpoint loaded from {path}")
				print(f"Resuming from epoch {checkpoint['epoch'] + 1}")

			return (
				checkpoint['epoch'],
				checkpoint['metrics'],  # Full metrics history
				checkpoint['hyperparameters'],
				checkpoint['metadata']
			)

		except Exception as e:
			print(f"Error loading checkpoint: {str(e)}")
			return -1, {}, {}, {}