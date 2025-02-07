import torch
import torch.nn as nn
from tqdm import tqdm

from .ModelMetrics import ModelMetrics
from .EarlyStoppingManager import EarlyStoppingManager
from .CheckpointManager import CheckpointManager


class UnifiedTrainer:
	"""
	A comprehensive trainer class for machine learning models with advanced training features.

	This trainer supports:
	- Weighted loss for imbalanced classes
	- Learning rate scheduling
	- Early stopping
	- Gradient clipping
	- Checkpoint management
	- Detailed metrics tracking

	Attributes:
		model (nn.Module): The neural network model to be trained
		optimizer (torch.optim.Optimizer): Optimization algorithm
		criterion (nn.Module): Loss function, defaults to weighted BCE loss
		device (str): Computing device (cuda/cpu)
		checkpoint_manager (CheckpointManager): Manages model checkpoints
		early_stopping (EarlyStoppingManager): Handles early stopping logic
	"""
	def __init__(
		self,
		model,
		optimizer,
		class_counts,
		criterion=None,
		scheduler=None,
		device='cuda',
		grad_clip=1.0,
		early_stopping_patience=3,
		save_name='model_checkpoint'
	):
		"""
		Initialize the UnifiedTrainer with model, optimization, and training parameters.

		Args:
			model (nn.Module): Neural network model to train
			optimizer (torch.optim.Optimizer): Optimization algorithm
			class_counts (tuple): Number of negative and positive class instances
			criterion (nn.Module, optional): Custom loss function
			scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler
			device (str, optional): Computing device. Defaults to 'cuda'
			grad_clip (float, optional): Gradient clipping threshold. Defaults to 1.0
			early_stopping_patience (int, optional): Epochs to wait for improvement. Defaults to 3
			save_name (str, optional): Base name for checkpoint files. Defaults to 'model_checkpoint'
		"""
		self.model = model
		self.optimizer = optimizer
		self.device = device

		# Set up weighted loss for imbalanced classes
		num_negatives, num_positives = class_counts
		pos_weight = num_negatives / num_positives
		pos_weight = torch.tensor([pos_weight], dtype=torch.float).to(self.device)
		self.criterion = criterion if criterion else nn.BCEWithLogitsLoss(pos_weight=pos_weight)
		
		self.scheduler = scheduler
		self.grad_clip = grad_clip
		self.last_lr = None 
		self.model_metrics = ModelMetrics()

		# Single checkpoint manager instance for both best and final saves
		self.checkpoint_manager = CheckpointManager(
			save_dir='checkpoints',
			save_name=save_name,
			verbose=True
		)
		
		self.early_stopping = EarlyStoppingManager(
			checkpoint_manager=self.checkpoint_manager,
			patience=early_stopping_patience,
			delta=1e-4,
			metric_name='val_loss'
		)

	def get_lr(self):
		"""
		Retrieve the current learning rate.

		Returns:
			float: Current learning rate of the first parameter group
		"""
		return self.optimizer.param_groups[0]['lr']

	def _forward_pass(self, embeddings):
		"""
		Perform a forward pass through the model, handling models with optional attention.

		Args:
			embeddings (torch.Tensor): Input embeddings/features

		Returns:
			torch.Tensor: Model predictions
		"""
		if hasattr(self.model, 'attention'):
			predictions, _ = self.model(embeddings)
		else:
			predictions = self.model(embeddings)
		return predictions

	def _process_batch(self, batch):
		"""
		Prepare batch data for training/evaluation.

		Args:
			batch (dict): Batch containing embeddings and labels

		Returns:
			tuple: Processed embeddings and labels tensors
		"""
		embeddings = batch['embeddings'].to(self.device)
		labels = batch['label'].to(self.device).unsqueeze(1)
		return embeddings, labels

	def train(self, train_loader, val_loader, epochs=10):
		"""
		Train the model with early stopping and periodic validation.

		Args:
			train_loader (DataLoader): Training data loader
			val_loader (DataLoader): Validation data loader
			epochs (int, optional): Maximum number of training epochs. Defaults to 10

		Returns:
			dict: Training metrics history
		"""
		early_stopped = False
		final_val_loss = float('inf')
		final_epoch = 0
		
		for epoch in range(epochs):
			# Training phase
			train_loss, train_preds, train_labels, train_proba_preds = self._train_epoch(
				train_loader, epoch, epochs
			)
			
			# Validation phase
			val_loss, val_metrics = self.evaluate(val_loader)
			final_val_loss = val_loss  # Store the last validation loss
			final_epoch = epoch
			
			# Calculate and update metrics
			train_metrics = self.model_metrics.calculate_metrics(
				train_labels, train_preds, train_proba_preds
			)
			self.model_metrics.update_metrics_history(
				epoch, train_loss / len(train_loader), train_metrics, val_loss, val_metrics
			)
			
			# Handle early stopping and scheduling
			early_stop = self._handle_scheduling_and_stopping(val_loss, epoch)
			
			# Print epoch metrics
			self.model_metrics.print_epoch_metrics(
				epoch, epochs, train_loss/len(train_loader), val_loss, val_metrics
			)
			
			if early_stop:
				print("Early stopping triggered")
				early_stopped = True
				break
		
		# Save final model state
		print("\nSaving final model state...")
		final_save_path = self.checkpoint_manager.save_checkpoint(
			model=self.model,
			optimizer=self.optimizer,
			scheduler=self.scheduler,
			epoch=final_epoch,
			metrics=self.model_metrics.metrics_history,
			hyperparameters={
				'learning_rate': self.get_lr(),
				'grad_clip': self.grad_clip,
				'early_stopping_patience': self.early_stopping.patience,
				'early_stopped': early_stopped,
				'final_val_loss': final_val_loss
			},
			early_stopping_state={
				'best_metric': self.early_stopping.best_metric,
				'counter': self.early_stopping.counter,
				'early_stop': self.early_stopping.early_stop
			},
			save_type='final'  # Explicitly save as final checkpoint
		)
		print(f"Final model saved at: {final_save_path}")
			
		return self.model_metrics.metrics_history

	def _train_epoch(self, train_loader, epoch, epochs):
		"""
		Perform a single training epoch.

		Args:
			train_loader (DataLoader): Training data loader
			epoch (int): Current epoch number
			epochs (int): Total number of epochs

		Returns:
			tuple: Training loss, predictions, labels, and probability predictions
		"""
		self.model.train()
		train_loss = 0
		train_preds, train_labels = [], []
		train_proba_preds = []
		
		progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training", colour="green")
		
		for batch in progress_bar:
			embeddings, labels = self._process_batch(batch)
			loss, proba_preds, preds = self._train_step(embeddings, labels)
			
			train_loss += loss.item()
			progress_bar.set_postfix({'loss': loss.item()})
			
			train_proba_preds.extend(proba_preds)
			train_preds.extend(preds)
			train_labels.extend(labels.cpu().numpy())
			
		return train_loss, train_preds, train_labels, train_proba_preds

	def _train_step(self, embeddings, labels):
		"""
		Perform a single training step with loss computation and parameter updates.

		Args:
			embeddings (torch.Tensor): Input feature embeddings
			labels (torch.Tensor): Target labels

		Returns:
			tuple: Loss, probability predictions, and binary predictions
		"""
		self.optimizer.zero_grad()
		predictions = self._forward_pass(embeddings)
		loss = self.criterion(predictions, labels.float())
		
		loss.backward()
		if self.grad_clip:
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
			
		self.optimizer.step()
		
		proba_preds = torch.sigmoid(predictions).cpu().detach().numpy()
		preds = (proba_preds > 0.5).astype(int)
		
		return loss, proba_preds, preds

	def _handle_scheduling_and_stopping(self, val_loss, current_epoch):
		"""
		Handle learning rate scheduling and early stopping logic.

		Args:
			val_loss (float): Current validation loss
			current_epoch (int): Current training epoch

		Returns:
			bool: Whether to stop training early
		"""
		# Handle learning rate scheduling
		current_lr = self.get_lr()
		if self.scheduler:
			if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
				self.scheduler.step(val_loss)
			elif isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
				self.scheduler.step()
				
		new_lr = self.get_lr()
		if new_lr != current_lr:
			print(f"\nLearning rate changed: {current_lr:.6f} -> {new_lr:.6f}")
			self.last_lr = new_lr
		
		# Check early stopping conditions
		should_stop, _ = self.early_stopping.check_early_stopping(
			current_metric=val_loss,
			model=self.model,
			optimizer=self.optimizer,
			scheduler=self.scheduler,
			epoch=current_epoch,
			metrics=self.model_metrics.metrics_history,
			hyperparameters={
				'learning_rate': self.get_lr(),
				'grad_clip': self.grad_clip,
				'early_stopping_patience': self.early_stopping.patience,
			}
		)

		return should_stop

	def evaluate(self, data_loader):
		"""
		Evaluate the model on a given data loader.

		Args:
			data_loader (DataLoader): Data loader for evaluation

		Returns:
			tuple: Evaluation loss and metrics dictionary
		"""
		self.model.eval()
		epoch_loss = 0
		all_preds, all_labels = [], []
		all_proba_preds = []
		
		with torch.no_grad():
			progress_bar = tqdm(data_loader, desc="Evaluating", colour="blue")
			for batch in progress_bar:
				embeddings, labels = self._process_batch(batch)
				predictions = self._forward_pass(embeddings)
				loss = self.criterion(predictions, labels.float())
				
				epoch_loss += loss.item()
				proba_preds = torch.sigmoid(predictions).cpu().detach().numpy()
				preds = (proba_preds > 0.5).astype(int)
				
				all_proba_preds.extend(proba_preds)
				all_preds.extend(preds)
				all_labels.extend(labels.cpu().numpy())
				
		metrics = self.model_metrics.calculate_metrics(all_labels, all_preds, all_proba_preds)
		metrics['loss'] = epoch_loss / len(data_loader)
		
		return metrics['loss'], metrics

	def test(self, test_loader, checkpoint_type='best'):
		"""
		Test the model using either the best or final checkpoint.

		Args:
			test_loader (DataLoader): Data loader for testing
			checkpoint_type (str, optional): Checkpoint to load. Defaults to 'best'

		Returns:
			dict: Test metrics including loss
		"""
		print(f"\nTesting with {checkpoint_type} model checkpoint...")
		
		# Load the specified checkpoint
		epoch, metrics, hyperparams, metadata, early_stopping_state = (
			self.checkpoint_manager.load_checkpoint(
				model=self.model,
				optimizer=self.optimizer,
				scheduler=self.scheduler,
				load_type=checkpoint_type
			)
		)
		
		test_loss, test_metrics = self.evaluate(test_loader)
		self.model_metrics.print_test_results(test_loss, test_metrics)
		
		test_metrics['loss'] = test_loss
		return test_metrics