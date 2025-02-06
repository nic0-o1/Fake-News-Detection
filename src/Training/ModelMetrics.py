import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve,average_precision_score,classification_report)

class ModelMetrics:
	"""A class for calculating and maintaining model evaluation metrics."""
	def __init__(self):
		"""
		Initialize the metrics class with a default metrics history structure.
		"""
		self.metrics_history = {
            'loss': {'train': [], 'val': []},
            'accuracy': {'train': [], 'val': []},
            'precision': {'train': [], 'val': []},
            'recall': {'train': [], 'val': []},
            'f1_score': {'train': [], 'val': []},
            'auc_roc': {'train': [], 'val': []},
            'confusion_matrix': {'train': [], 'val': []},
            'precision_recall_curve': {'train': [], 'val': []},
            'roc_curve': {'train': [], 'val': []}
        }

	def calculate_metrics(self,labels, preds, proba_preds=None):
		"""
		Calculate evaluation metrics for a given set of predictions and labels.

		Args:
			labels (list or np.array): True labels.
			preds (list or np.array): Predicted labels (binary).
			proba_preds (list or np.array): Predicted probabilities (for AUC-ROC and curve metrics).

		Returns:
			dict: A dictionary of calculated metrics.
		"""
		metrics = {
			'accuracy': accuracy_score(labels, preds),
			'precision': precision_score(labels, preds, zero_division=0),
			'recall': recall_score(labels, preds, zero_division=0),
			'f1_score': f1_score(labels, preds, zero_division=0),
			'confusion_matrix': confusion_matrix(labels, preds)
		}

		if proba_preds is not None:
			metrics['auc_roc'] = roc_auc_score(labels, proba_preds)
			metrics['roc_curve'] = roc_curve(labels, proba_preds)
			metrics['precision_recall_curve'] = precision_recall_curve(labels, proba_preds)
			metrics['average_precision'] = average_precision_score(labels, proba_preds)
		
		metrics['classification_report'] = classification_report(labels, preds, output_dict=True)

		return metrics

	def update_metrics_history(self, epoch, train_loss, train_metrics, val_loss, val_metrics):
		"""
		Update the metrics history with new training and validation metrics.

		Args:
			epoch (int): Current epoch number.
			train_loss (float): Training loss.
			train_metrics (dict): Training metrics (accuracy, precision, etc.).
			val_loss (float): Validation loss.
			val_metrics (dict): Validation metrics (accuracy, precision, etc.).
		"""
		# Update loss
		self.metrics_history['loss']['train'].append(train_loss)
		self.metrics_history['loss']['val'].append(val_loss)

		# Update other metrics
		metrics_to_update = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
		for metric in metrics_to_update:
			if metric in train_metrics:
				self.metrics_history[metric]['train'].append(train_metrics[metric])
			if metric in val_metrics:
				self.metrics_history[metric]['val'].append(val_metrics[metric])

		# Update confusion matrix, roc_curve, and precision_recall_curve for validation only
		for metric in ['confusion_matrix', 'roc_curve', 'precision_recall_curve']:
			if metric in val_metrics:
				self.metrics_history[metric]['val'].append(val_metrics[metric])

	def print_epoch_metrics(self, epoch, epochs, train_loss, val_loss, val_metrics):
		"""
		Print the metrics for the current epoch.

		Args:
			epoch (int): Current epoch number.
			epochs (int): Total number of epochs.
			train_loss (float): Training loss.
			val_loss (float): Validation loss.
			val_metrics (dict): Validation metrics (accuracy, f1_score, etc.).
		"""
		print(f"Epoch {epoch + 1}/{epochs}")
		print(f" Train Loss: {train_loss:.4f}")
		print(f" Val Loss: {val_loss:.4f}")
		print(f" Val Accuracy: {val_metrics.get('accuracy', 0):.4f}")
		print(f" Val F1 Score: {val_metrics.get('f1_score', 0):.4f}")
		print(f" Precision: {val_metrics.get('precision'):.4f}")
		print(f" Recall: {val_metrics.get('recall'):.4f}")
		if 'auc_roc' in val_metrics:
			print(f" Val AUC-ROC: {val_metrics['auc_roc']:.4f}")
		print()
		
	def print_test_results(self, test_loss, test_metrics):
		"""
		Print the test results after evaluation.

		Args:
			test_loss (float): Test loss.
			test_metrics (dict): Test metrics (accuracy, f1_score, etc.).
		"""
		print("\nTest Results:")
		print(f"Loss: {test_loss:.4f}")
		print(f"Accuracy: {test_metrics['accuracy']:.4f}")
		print(f"F1 Score: {test_metrics['f1_score']:.4f}")
		print(f"Precision: {test_metrics['precision']:.4f}")
		print(f"Recall: {test_metrics['recall']:.4f}")
		print(f"ROC AUC: {test_metrics['auc_roc']:.4f}")
		print()