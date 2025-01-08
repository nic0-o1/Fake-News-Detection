import torch
from typing import Optional

class EarlyStopping:
	def __init__(self, patience: int = 3, delta: float = 0.0, path: str = 'checkpoint.pt', verbose: bool = True):
		"""
		Initialize the EarlyStopping object.

		Args:
			patience (int): Number of epochs with no improvement before stopping.
			delta (float): Minimum improvement to qualify as an improvement.
			path (str): Path to save the checkpoint file.
			verbose (bool): Whether to log checkpoint saving details.
		"""
		self.patience = patience
		self.delta = delta
		self.path = path
		self.verbose = verbose
		self.best_loss: Optional[float] = None
		self.counter: int = 0
		self.early_stop: bool = False

	def __call__(self, val_loss: float, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> None:
		"""
		Check if training should stop and save the training state if validation loss improves.

		Args:
			val_loss (float): Current validation loss.
			model (torch.nn.Module): PyTorch model being trained.
			optimizer (torch.optim.Optimizer): Optimizer used for training.
			epoch (int): Current epoch number.
			scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler, if used.
		"""
		if self.best_loss is None:
			self.best_loss = val_loss
			self.save_checkpoint(model, optimizer, epoch, scheduler)
		elif val_loss > self.best_loss + self.delta:
			self.counter += 1
			if self.verbose:
				print(f"Validation loss did not improve. Counter: {self.counter}/{self.patience}")
			if self.counter >= self.patience:
				self.early_stop = True
		else:
			self.best_loss = val_loss
			self.save_checkpoint(model, optimizer, epoch, scheduler)
			self.counter = 0

	def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> None:
		"""
		Save the training state when validation loss improves.

		Args:
			model (torch.nn.Module): PyTorch model being trained.
			optimizer (torch.optim.Optimizer): Optimizer used for training.
			epoch (int): Current epoch number.
			scheduler (torch.optim.lr_scheduler, optional): Scheduler to save state for, if used.
		"""
		checkpoint = {
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'epoch': epoch,
			'best_loss': self.best_loss,
			'counter': self.counter,
			'early_stop': self.early_stop,
		}
		if scheduler:
			checkpoint['scheduler_state_dict'] = scheduler.state_dict()
		
		torch.save(checkpoint, self.path)
		if self.verbose:
			print(f"Checkpoint saved to '{self.path}' at epoch {epoch} with validation loss {self.best_loss:.4f}.")

	def load_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> int:
		"""
		Load the training state from a checkpoint file.

		Args:
			model (torch.nn.Module): PyTorch model to load state into.
			optimizer (torch.optim.Optimizer): Optimizer to load state into.
			scheduler (torch.optim.lr_scheduler, optional): Scheduler to load state into.

		Returns:
			int: The epoch number to resume training from.
		"""
		checkpoint = torch.load(self.path)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		if scheduler and 'scheduler_state_dict' in checkpoint:
			scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
