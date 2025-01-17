import random

import json

import string
import numpy as np
import torch
import torch.optim as optim

from config import *
from Training.UnifiedTrainer import UnifiedTrainer
from Models.LSTMWithAttention import LSTMWithAttention
from Models.LSTMWithoutAttention import LSTMWithoutAttention

from Training.ModelMetricsVisualizer import ModelMetricsVisualizer
import matplotlib.pyplot as plt

def set_seeds(seed=42):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True

def visualize_results(results):
	"""
	Create and save publication-ready visualizations.
	
	Args:
		results: Dictionary containing results for both models
	"""
	visualizer = ModelMetricsVisualizer()
	
	metrics = ['accuracy', 'precision', 'recall', 'f1_score']
	for metric in metrics:
		visualizer.plot_metric(results, metric, f'{metric}_comparison')
	
	visualizer.plot_confusion_matrix(results, 'confusion_matrix_comparison')
	
	plt.show()

table = str.maketrans('', '', string.punctuation)

def makeWords(sentences):
  wordList = []
  for headline in sentences:
    words = headline.split(' ')
    stripped = [w.strip().translate(table) for w in words]
    wordList.append(stripped)
  return wordList
	
def init_LSTM_with_attention():
	return LSTMWithAttention(
		embedding_dim=EMBEDDING_DIM,
		hidden_dim=128,
		output_dim=1,
		dropout_rate=0.7
	)

def init_LSTM_without_attention():
	return LSTMWithoutAttention(
		embedding_dim=EMBEDDING_DIM,
		hidden_dim=128,
		output_dim=1,
		dropout_rate=0.7 # changed from 0.7 to 0.4
	)

def init_training_components(model, lr=0.00005, scheduler_factor = 0.25, scheduler_patience = 3):
	optimizer = optim.AdamW(model.parameters(), lr=lr)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(
		optimizer,
		mode='min',
		factor=scheduler_factor, #was 0.1
		patience=scheduler_patience
	)
	return optimizer, scheduler

def train_evaluate_and_test_models(class_counts, train_loader, val_loader, test_loader, epochs=10):
	
	# Set seeds for reproducibility
	set_seeds()

	# Initialize both models
	model1 = init_LSTM_with_attention().to(DEVICE)
	model2 = init_LSTM_without_attention().to(DEVICE)


	# Dictionary to store results
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
		save_name = 'LSTMWithAttention'
	)
	training_metrics1 = trainer1.train(train_loader, val_loader, epochs)
	test_metrics1 = trainer1.test(test_loader)

	results['Model 1']['trainer'] = trainer1
	results['Model 1']['training_metrics'] = training_metrics1
	results['Model 1']['test_metrics'] = test_metrics1
	# Reset seeds for consistent comparison
	set_seeds()

	#Train Model 2
	print("\nTraining Model 2 (LSTM without Attention)")

	optimizer2, scheduler2 = init_training_components(model = model2, lr=0.0001, scheduler_factor = 0.5)
	trainer2 = UnifiedTrainer(
		model=model2,
		optimizer=optimizer2,
		class_counts=class_counts,
		scheduler=scheduler2,
		device=DEVICE,
		grad_clip=1.0, # changed from 1.0 to 0.5
		early_stopping_patience=EARLY_STOP_PATIENCE,
		save_name = 'LSTMWithoutAttention'
	)
	
	training_metrics2 = trainer2.train(train_loader, val_loader, epochs)
	test_metrics2 = trainer2.test(test_loader)
	
	results['Model 2']['trainer'] = trainer2
	results['Model 2']['training_metrics'] = training_metrics2
	results['Model 2']['test_metrics'] = test_metrics2

	save_results(results)

	return results
def save_results(results):
	with open('Results/results.json', 'w') as f:
		json.dump(results, f, indent=4)

def load_results():
	with open('Results/results.json', 'r') as f:
		results = json.load(f)
	return results
def compare_models(results):
	print("\nModel Comparison:")
	print("-" * 50)

	# Compare training/validation metrics
	print("\nTraining/Validation Metrics:")
	metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score']
	for metric in metrics_to_compare:
		print(f"\n{metric.upper()}:")
		for model_name in results:
			val_scores = results[model_name]['training_metrics'][metric]['val']
			final_score = val_scores[-1]
			best_score = max(val_scores)

			print(f"{model_name}:")
			print(f"  Final: {final_score:.4f}")
			print(f"  Best:  {best_score:.4f}")
			print(f"  Mean:  {np.mean(val_scores):.4f}")
			print(f"  Std:   {np.std(val_scores):.4f}")
	
	# Compare test metrics
	print("\nTest Metrics:")
	for metric in metrics_to_compare:
		print(f"\n{metric.upper()}:")
		for model_name in results:
			test_score = results[model_name]['test_metrics'][metric]
			print(f"{model_name}: {test_score:.4f}")

