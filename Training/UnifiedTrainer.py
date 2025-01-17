import torch
import torch.nn as nn
from tqdm import tqdm

from .EarlyStoppingCheckpointManager import EarlyStoppingCheckpointManager
from .ModelMetrics import ModelMetrics


class UnifiedTrainer:
	def __init__(self, model, optimizer, class_counts, criterion=None, scheduler=None, device='cuda', grad_clip=1.0, early_stopping_patience=2, save_name='model_checkpoint.pt'):
		self.model = model
		self.optimizer = optimizer
		self.device = device

		num_negatives, num_positives = class_counts
		pos_weight = num_negatives / num_positives
		pos_weight = torch.tensor([pos_weight], dtype=torch.float).to(self.device)

		self.criterion = criterion if criterion else nn.BCEWithLogitsLoss(pos_weight=pos_weight)
		
		self.scheduler = scheduler
		self.grad_clip = grad_clip
		self.last_lr = None 
		self.model_metrics = ModelMetrics()

		self.checkpoint_manager = EarlyStoppingCheckpointManager(
			save_dir='checkpoints',
			save_name=save_name,
			patience=early_stopping_patience,
			delta=1e-4,
			verbose=True,
			metric_name='val_loss',
		)

	def get_lr(self):
		return self.optimizer.param_groups[0]['lr']

	def _forward_pass(self, embeddings):
		if hasattr(self.model, 'attention'):
			predictions, _ = self.model(embeddings)
		else:
			predictions = self.model(embeddings)
		return predictions

	def _process_batch(self, batch):
		embeddings = batch['embeddings'].to(self.device)
		labels = batch['label'].to(self.device).unsqueeze(1)
		return embeddings, labels

	def train(self, train_loader, val_loader, epochs=10):
		for epoch in range(epochs):
			train_loss, train_preds, train_labels, train_proba_preds = self._train_epoch(train_loader, epoch, epochs)
			val_loss, val_metrics = self.evaluate(val_loader)
			train_metrics = self.model_metrics.calculate_metrics(train_labels, train_preds, train_proba_preds)
			self.model_metrics.update_metrics_history(epoch, train_loss / len(train_loader), train_metrics, val_loss, val_metrics)
			
			early_stop = self._handle_scheduling_and_stopping(val_loss, epoch)
			
			self.model_metrics.print_epoch_metrics(epoch, epochs, train_loss/len(train_loader), val_loss, val_metrics)
			if early_stop:
				break
			
		return self.model_metrics.metrics_history

	def _train_epoch(self, train_loader, epoch, epochs):
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

	def _handle_scheduling_and_stopping(self, val_loss,current_epoch):
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
		
		should_stop, _ = self.checkpoint_manager.check_early_stopping(
			current_metric=val_loss,
			model=self.model,
			optimizer=self.optimizer,
			scheduler=self.scheduler,
			epoch=current_epoch,
			metrics=self.model_metrics.metrics_history,
			hyperparameters={
				'learning_rate': self.get_lr(),
				'grad_clip': self.grad_clip,
				'early_stopping_patience': self.checkpoint_manager.patience,
			}
		)

		return should_stop

	def evaluate(self, data_loader):
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


	def test(self, test_loader):
		print("\nTesting with best model from checkpoint...")
		# Load the best model checkpoint before testing
		self.checkpoint_manager.load_checkpoint(
			model=self.model,
			optimizer=self.optimizer,
			scheduler=self.scheduler
		)
		test_loss, test_metrics = self.evaluate(test_loader)

		self.model_metrics.print_test_results(test_loss, test_metrics)
		test_metrics['loss'] = test_loss
		return test_metrics