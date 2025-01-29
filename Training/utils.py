import random
import json
import string
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from config import *
from Training.UnifiedTrainer import UnifiedTrainer
from Models.LSTMWithAttention import LSTMWithAttention
from Models.LSTMWithoutAttention import LSTMWithoutAttention
from Training.ModelMetricsVisualizer import ModelMetricsVisualizer

# Set up a translation table to remove punctuation from strings.
TABLE = str.maketrans('', '', string.punctuation)

def set_seeds(seed: int = 42) -> None:
	"""
	Set seeds for reproducibility across numpy, random, and torch.
	"""
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True

def visualize_results(metrics: list, results: dict) -> None:
	"""
	Create and display visualizations for model metrics.

	Args:
		results (dict): Dictionary containing results for models.
	"""
	visualizer = ModelMetricsVisualizer()
	

	for metric in metrics:
		visualizer.plot_metric(results, metric, f'{metric}_comparison')

	visualizer.plot_confusion_matrix(results, 'confusion_matrix_comparison')

	visualizer.plot_auc_roc_curve(results, 'auc_roc_curve_comparison')

	visualizer.plot_precision_recall_curve(results, 'precision_recall_curve_comparison')
	
	plt.show()

def make_words(sentences: list[str]) -> list[list[str]]:
	"""
	Tokenize sentences into words and remove punctuation.

	Args:
		sentences (list of str): List of sentences.

	Returns:
		list of list of str: Tokenized and cleaned words.
	"""
	return [[word.strip().translate(TABLE) for word in sentence.split(' ')] for sentence in sentences]

def init_lstm_with_attention() -> LSTMWithAttention:
	"""Initialize the LSTM model with attention."""
	return LSTMWithAttention(
		embedding_dim=EMBEDDING_DIM,
		hidden_dim=128,
		output_dim=1,
		dropout_rate=0.4
	)

def init_lstm_without_attention() -> LSTMWithoutAttention:
	"""Initialize the LSTM model without attention."""
	return LSTMWithoutAttention(
		embedding_dim=EMBEDDING_DIM,
		hidden_dim=128,
		output_dim=1,
		dropout_rate=0.4
	)

def init_training_components(
	model: torch.nn.Module, 
	lr: float = 0.00005, 
	scheduler_factor: float = 0.25, 
	scheduler_patience: int = 3
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
	"""
	Initialize optimizer and learning rate scheduler.

	Args:
		model (torch.nn.Module): The model to optimize.
		lr (float): Learning rate.
		scheduler_factor (float): Factor by which the learning rate is reduced.
		scheduler_patience (int): Number of epochs with no improvement before reducing the learning rate.

	Returns:
		tuple: Optimizer and scheduler.
	"""
	optimizer = optim.AdamW(model.parameters(), lr=lr)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(
		optimizer,
		mode='min',
		factor=scheduler_factor,
		patience=scheduler_patience
	)
	return optimizer, scheduler

def train_evaluate_and_test_models(
	class_counts: dict, 
	train_loader: torch.utils.data.DataLoader, 
	val_loader: torch.utils.data.DataLoader, 
	test_loader: torch.utils.data.DataLoader, 
	epochs: int = 10
) -> dict:
	"""
	Train, evaluate, and test two LSTM models (with and without attention).

	Args:
		class_counts (dict): Class distribution for weighted loss.
		train_loader (DataLoader): Training data loader.
		val_loader (DataLoader): Validation data loader.
		test_loader (DataLoader): Test data loader.
		epochs (int): Number of training epochs.

	Returns:
		dict: Results containing metrics for both models.
	"""
	set_seeds()

	# Initialize models
	model1 = init_lstm_with_attention().to(DEVICE)
	model2 = init_lstm_without_attention().to(DEVICE)

	# Store results
	results = {
		'Model 1': {'trainer': None, 'training_metrics': None, 'test_metrics': None},
		'Model 2': {'trainer': None, 'training_metrics': None, 'test_metrics': None}
	}

	# Train Model 1
	print("\nTraining Model 1 (LSTM with Attention)")
	optimizer1, scheduler1 = init_training_components(model1)
	trainer1 = UnifiedTrainer(
		model=model1,
		optimizer=optimizer1,
		class_counts=class_counts,
		scheduler=scheduler1,
		device=DEVICE,
		grad_clip=1.0,
		early_stopping_patience=EARLY_STOP_PATIENCE,
		save_name='LSTMWithAttention'
	)
	results['Model 1']['training_metrics'] = trainer1.train(train_loader, val_loader, epochs)
	results['Model 1']['test_metrics'] = trainer1.test(test_loader)

	set_seeds()  # Reset seeds for consistency

	# Train Model 2
	print("\nTraining Model 2 (LSTM without Attention)")
	optimizer2, scheduler2 = init_training_components(model2, lr=0.0001, scheduler_factor=0.5)
	trainer2 = UnifiedTrainer(
		model=model2,
		optimizer=optimizer2,
		class_counts=class_counts,
		scheduler=scheduler2,
		device=DEVICE,
		grad_clip=0.5,
		early_stopping_patience=EARLY_STOP_PATIENCE,
		save_name='LSTMWithoutAttention'
	)
	results['Model 2']['training_metrics'] = trainer2.train(train_loader, val_loader, epochs)
	results['Model 2']['test_metrics'] = trainer2.test(test_loader)

	save_results(results)
	print('Model training and testing complete.')
	return results

def save_results(results: dict, filepath: str = 'checkpoints/results.json') -> None:
	"""
	Save training and testing results to a JSON file.

	Args:
		results (dict): Results to save.
		filepath (str): Filepath for saving the results.
	"""
	def convert_to_serializable(obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		elif isinstance(obj, dict):
			return {key: convert_to_serializable(value) for key, value in obj.items()}
		elif isinstance(obj, list):
			return [convert_to_serializable(item) for item in obj]
		elif isinstance(obj, tuple):
			return tuple(convert_to_serializable(item) for item in obj)
		return obj

	results_to_save = {
		model_name: {
			key: convert_to_serializable(value)
			for key, value in metrics.items() if key != 'trainer'
		}
		for model_name, metrics in results.items()
	}

	with open(filepath, 'w') as f:
		json.dump(results_to_save, f, indent=4)

def load_results(filepath: str = 'Results/results.json') -> dict | None:
	"""
	Load results from a JSON file.

	Args:
		filepath (str): Filepath to load results from.

	Returns:
		dict or None: Loaded results, or None if file is not found.
	"""
	try:
		with open(filepath, 'r') as f:
			return json.load(f)
	except FileNotFoundError:
		print(f"No results file found at {filepath}")
		return None

def compare_models(results: dict) -> None:
	"""
	Compare models based on training, validation, and test metrics.

	Args:
		results (dict): Results containing metrics for models.
	"""
	print("\nModel Comparison:")
	print("-" * 50)

	metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score']

	print("\nTraining/Validation Metrics:")
	for metric in metrics_to_compare:
		print(f"\n{metric.upper()}:")
		for model_name, metrics in results.items():
			val_scores = metrics['training_metrics'][metric]['val']
			print(f"{model_name}:")
			print(f"  Final: {val_scores[-1]:.4f}")
			print(f"  Best:  {max(val_scores):.4f}")
			print(f"  Mean:  {np.mean(val_scores):.4f}")
			print(f"  Std:   {np.std(val_scores):.4f}")

	print("\nTest Metrics:")
	for metric in metrics_to_compare:
		print(f"\n{metric.upper()}:")
		for model_name, metrics in results.items():
			print(f"{model_name}: {metrics['test_metrics'][metric]:.4f}")
