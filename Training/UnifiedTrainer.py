import torch
import torch.nn as nn
from tqdm import tqdm

from .EarlyStopping import EarlyStopping
from .ModelMetrics import ModelMetrics


class UnifiedTrainer:
	"""A unified trainer class for handling different model architectures with various training utilities."""

	def __init__(
		self, model, optimizer, criterion=None, scheduler=None, device='cuda', grad_clip=1.0, early_stopping_patience=2, save_name='model_checkpoint.pt'):
		"""
		Initialize the trainer with model and training parameters.

		Args:
			model: The neural network model to train
			optimizer: The optimizer to use for training
			criterion: Loss function (defaults to BCEWithLogitsLoss)
			scheduler: Learning rate scheduler (optional)
			device: Device to run the model on ('cuda' or 'cpu')
			grad_clip: Maximum gradient norm for clipping
			early_stopping_patience: Number of epochs to wait before early stopping
			save_name: Path to save the best model checkpoint
		"""
		self.model = model
		self.optimizer = optimizer
		self.criterion = criterion if criterion else nn.BCEWithLogitsLoss()
		self.scheduler = scheduler
		self.device = device
		self.grad_clip = grad_clip
		self.early_stopping = EarlyStopping(patience=early_stopping_patience, path=save_name)
		self.metrics_history = ModelMetrics().DEFAULT_METRICS.copy()
		self.model_metrics = ModelMetrics()

	def _forward_pass(self, embeddings):
		"""Perform forward pass handling different model architectures."""
		if hasattr(self.model, 'attention'):
			predictions, _ = self.model(embeddings)
		else:
			predictions = self.model(embeddings)
		return predictions

	def _process_batch(self, batch):
		"""Process a single batch of data."""
		embeddings = batch['embeddings'].to(self.device)
		labels = batch['label'].to(self.device).unsqueeze(1)
		return embeddings, labels

	def train(self, train_loader, val_loader, epochs=10):
		"""
		Train the model for the specified number of epochs.

		Args:
			train_loader: DataLoader for training data
			val_loader: DataLoader for validation data
			epochs: Number of epochs to train for

		Returns:
			dict: Training history containing metrics
		"""
		for epoch in range(epochs):
			# Training phase
			train_loss, train_preds, train_labels, train_proba_preds = self._train_epoch(train_loader, epoch, epochs)
			
			# Validation phase
			val_loss, val_metrics = self.evaluate(val_loader)
			
			# Update metrics and handle scheduling/early stopping
			train_metrics = self.model_metrics.calculate_metrics(train_labels, train_preds, train_proba_preds)
			self.model_metrics.update_metrics_history(
				epoch, train_loss / len(train_loader), train_metrics, val_loss, val_metrics
			)
			
			if self._handle_scheduling_and_stopping(val_loss):
				self.print_epoch_metrics(epoch, epochs, train_loss/len(train_loader),val_loss, val_metrics)
				break
				
			self.model_metrics.print_epoch_metrics(epoch, epochs, train_loss/len(train_loader), val_loss, val_metrics)
			
		return self.metrics_history

	def _train_epoch(self, train_loader, epoch, epochs):
		"""Train for one epoch and return loss and predictions."""
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
			
		return train_loss, train_preds, train_labels,train_proba_preds

	def _train_step(self, embeddings, labels):
		"""Perform a single training step."""
		self.optimizer.zero_grad()
		predictions = self._forward_pass(embeddings)
		loss = self.criterion(predictions, labels.float())
		loss.backward()
		
		if self.grad_clip:
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
			
		self.optimizer.step()
		# preds = (torch.sigmoid(predictions) > 0.5).float().cpu().numpy()
		
		proba_preds = torch.sigmoid(predictions).cpu().detach().numpy()
		# For binary classification, the threshold can still be applied here if needed.
		preds = (proba_preds > 0.5).astype(int)
		
		return loss, proba_preds, preds

	def _handle_scheduling_and_stopping(self, val_loss):
		"""Handle learning rate scheduling and early stopping."""
		if self.scheduler:
			if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
				self.scheduler.step(val_loss)
			else:
				self.scheduler.step()

		self.early_stopping(val_loss, self.model, self.optimizer, self.scheduler)
		return self.early_stopping.early_stop

	def evaluate(self, data_loader):
		"""Evaluate the model on a data loader."""
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
				# preds = (torch.sigmoid(predictions) > 0.5).cpu().numpy()
				proba_preds = torch.sigmoid(predictions).cpu().detach().numpy()
				preds = (proba_preds > 0.5).astype(int)
				
				all_proba_preds.extend(proba_preds)
				all_preds.extend(preds)
				all_labels.extend(labels.cpu().numpy())

		metrics = self.model_metrics.calculate_metrics(all_labels, all_preds,all_proba_preds)
		metrics['loss'] = epoch_loss / len(data_loader)
		
		return metrics['loss'], metrics
	
	def test(self, test_loader):
		"""
		Test the model on the test dataset using the best saved model.
		
		Args:
			test_loader: DataLoader for test data
			
		Returns:
			dict: Test metrics
		"""

		print("\nTesting with best model from checkpoint...")
		test_loss, test_metrics = self.evaluate(test_loader)

		print("\nTest Results:")
		print(f"Loss: {test_loss:.4f}")  
		print(f"Accuracy: {test_metrics['accuracy']:.4f}")
		print(f"F1 Score: {test_metrics['f1_score']:.4f}")
		print(f"Precision: {test_metrics['precision']:.4f}")
		print(f"Recall: {test_metrics['recall']:.4f}")
		print(f"ROC AUC: {test_metrics['roc_auc']:.4f}")
		
		# Add the test loss to the metrics dictionary before returning
		test_metrics['loss'] = test_loss  # Ensure loss is included in returned metrics

		return test_metrics
	
def print_epoch_metrics(self, epoch, epochs, train_loss, val_metrics):
		"""Print the metrics for the current epoch."""
		print(f"Epoch {epoch+1}/{epochs}")
		print(f" Train Loss: {train_loss:.4f}")
		print(f" Val Loss: {val_metrics['loss']:.4f}")
		print(f" Val Accuracy: {val_metrics['accuracy']:.4f}")
		print(f" Val F1 Score: {val_metrics['f1_score']:.4f}")
		print(f" Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}\n")