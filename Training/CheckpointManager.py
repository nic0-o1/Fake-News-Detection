import torch
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

class CheckpointManager:
	def __init__(self, save_dir: str, save_name: str = 'Model', verbose: bool = True):
		self.save_dir = Path(save_dir)
		self.save_dir.mkdir(parents=True, exist_ok=True)
		self.base_save_name = save_name
		self.verbose = verbose
		
		# Create paths for different types of saves
		self._update_save_paths()
	
	def _update_save_paths(self):
		"""Update all save paths based on base_save_name"""
		self.best_path = self.save_dir / f"{self.base_save_name}_best.pt"
		self.final_path = self.save_dir / f"{self.base_save_name}_final.pt"
	
	def _create_metadata(self, epoch: int, metrics: dict, hyperparameters: dict) -> dict:
		return {
			'timestamp': datetime.now().isoformat(),
			'epoch': epoch,
			'metrics': metrics,
			'hyperparameters': hyperparameters,
			'pytorch_version': torch.__version__
		}
	
	def save_checkpoint(
		self,
		model: torch.nn.Module,
		optimizer: torch.optim.Optimizer,
		scheduler: Optional[Any],
		epoch: int,
		metrics: dict,
		hyperparameters: dict,
		early_stopping_state: dict,
		save_type: str = 'best'
	) -> Optional[str]:
		"""
		Save a checkpoint with specified type.
		
		Args:
			save_type: Either 'best' or 'final'
		"""
		try:
			checkpoint = {
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
				'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
				'epoch': epoch,
				'metrics': metrics,
				'hyperparameters': hyperparameters,
				'early_stopping_state': early_stopping_state,
				'metadata': self._create_metadata(epoch, metrics, hyperparameters)
			}
			
			save_path = self.best_path if save_type == 'best' else self.final_path
			torch.save(checkpoint, save_path)
			
			if self.verbose:
				print(f"{save_type.capitalize()} checkpoint saved at {save_path}")
			
			return str(save_path)
			
		except Exception as e:
			print(f"Error saving {save_type} checkpoint: {str(e)}")
			return None
	
	def load_checkpoint(
		self,
		model: torch.nn.Module,
		optimizer: torch.optim.Optimizer,
		scheduler: Optional[Any] = None,
		load_type: str = 'best'
	) -> Tuple[int, Dict, Dict, Dict, Dict]:
		"""
		Load a checkpoint of specified type.
		
		Args:
			load_type: Either 'best' or 'final'
		"""
		# try:
		load_path = self.best_path if load_type == 'best' else self.final_path
		checkpoint = torch.load(str(load_path))
		
		# Verify checkpoint integrity
		required_keys = {'model_state_dict', 'epoch', 'early_stopping_state'}
		if not all(key in checkpoint for key in required_keys):
			raise ValueError("Checkpoint is missing required keys")
		
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		
		if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
			scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		
		if self.verbose:
			print(f"Checkpoint loaded from {load_path}")
			print(f"Resuming from epoch {checkpoint['epoch'] + 1}")
		
		return (
			checkpoint['epoch'],
			checkpoint['metrics'],
			checkpoint['hyperparameters'],
			checkpoint['metadata'],
			checkpoint['early_stopping_state']
		)
			
		# except Exception as e:
		#     print(f"Error loading {load_type} checkpoint: {str(e)}")
		#     return -1, {}, {}, {}, {}