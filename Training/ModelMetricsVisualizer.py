import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Optional
from pathlib import Path

class ModelMetricsVisualizer:
	"""
	A class for visualizing model metrics in a publication-ready format.
	"""
	
	def __init__(self):
		"""Initialize the visualizer with publication-friendly settings."""
		plt.rcParams.update({
			'font.family': 'serif',
			'font.size': 10,
			'axes.labelsize': 10,
			'axes.titlesize': 10,
			'xtick.labelsize': 9,
			'ytick.labelsize': 9,
			'legend.fontsize': 9,
			'figure.dpi': 300
		})
		
		self.colors = {
			'Model 1': '#0077BB',
			'Model 2': '#EE3377',
		}
		
		self.line_styles = {
			'train': (0, (1, 1)),
			'val': 'solid',
			'test': (0, (5, 1))
		}
		
		self.save_dir = Path('Results')
		self.save_dir.mkdir(parents=True, exist_ok=True)
	
	def plot_metric(self, results: Dict, metric: str, save_path: Optional[str] = None) -> plt.Figure:
		"""
		Create a publication-ready plot for a specific metric.
		
		Args:
			results: Dictionary containing results for both models
			metric: The metric to plot (accuracy, precision, recall, or f1_score)
			save_path: Optional path to save the plot
		"""
		fig, ax = plt.subplots(figsize=(8, 5))
		
		for model_name in ['Model 1', 'Model 2']:
			epochs = range(1, len(results[model_name]['training_metrics'][metric]['val']) + 1)
			train_scores = results[model_name]['training_metrics'][metric]['train']
			val_scores = results[model_name]['training_metrics'][metric]['val']
			test_score = results[model_name]['test_metrics'][metric]
			
			# Training and validation curves
			ax.plot(epochs, train_scores, 
				   linestyle=self.line_styles['train'],
				   color=self.colors[model_name],
				   label=f'{model_name} (Train)',
				   linewidth=1.5)
			
			ax.plot(epochs, val_scores,
				   linestyle=self.line_styles['val'],
				   color=self.colors[model_name],
				   label=f'{model_name} (Val)',
				   linewidth=1.5)
			
			# Test score line
			ax.axhline(y=test_score,
					  linestyle=self.line_styles['test'],
					  color=self.colors[model_name],
					  label=f'{model_name} (Test)',
					  linewidth=1, alpha=0.7)

		
		ax.set_xlabel('Epoch')
		ax.set_ylabel(metric.replace('_', ' ').title())
		ax.set_ylim(0, 1)
		ax.set_xlim(0, len(epochs) + 1)
		
		ax.grid(True, linestyle=':', alpha=0.3)
		
		ax.legend(loc='center right',
				 bbox_to_anchor=(0.98, 0.5),
				 frameon=True,
				 fancybox=True,
				 framealpha=0.9)
		
		plt.title(f'{metric.replace("_", " ").title()} Comparison')
		plt.tight_layout(pad=1.5)
		
		if save_path:
			plt.savefig(self.save_dir / f'{save_path}.png',
					   bbox_inches='tight',
					   dpi=300)
			
		return fig
	
	def plot_confusion_matrix(self, results: Dict, save_path: Optional[str] = None) -> plt.Figure:
		"""
		Create a publication-ready confusion matrix comparison.
		
		Args:
			results: Dictionary containing results for both models
			save_path: Optional path to save the plot
		"""
		figures = []
	
		for model_name, title in [('Model 1', 'LSTM with Attention'),
								('Model 2', 'LSTM without Attention')]:
			
			# Extract the confusion matrix
			cm = results[model_name]['test_metrics']['confusion_matrix']
			
			# Create a new figure
			fig, ax = plt.subplots(figsize=(6, 5))
			
			# Plot the heatmap
			sns.heatmap(cm, annot=True, fmt='d',
						cmap='Blues',
						xticklabels=['Negative', 'Positive'],
						yticklabels=['Negative', 'Positive'],
						cbar=True, ax=ax)
			
			# Set titles and labels
			ax.set_title(title, pad=10)
			ax.set_xlabel('Predicted')
			ax.set_ylabel('Actual')

			plt.tight_layout(pad=2.0, w_pad=3.0)
			
			# Save the figure if save_path is provided
			if save_path:
				fig.savefig(f"{self.save_dir /save_path}_{model_name.replace(' ', '_')}.png",
							bbox_inches='tight',
							dpi=300)
			
			# Append the figure to the list
			figures.append(fig)
		
		return figures
	def plot_auc_roc_curve(self,results: Dict, save_path: Optional[str] = None) -> plt.Figure:
		"""
		Create a publication-ready ROC curve comparison.
		
		Args:
			results: Dictionary containing results for both models
			save_path: Optional path to save the plot
		"""
		fig, ax = plt.subplots(figsize=(6, 5))
		
		for model_name in ['Model 1', 'Model 2']:

			fpr, tpr, _ = results[model_name]['test_metrics']['roc_curve']
			roc = results[model_name]['test_metrics']["auc_roc"]
			
			ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc:.3f})',
					color=self.colors[model_name],
					linewidth=1.5)
		
		ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
		
		ax.set_xlabel('False Positive Rate')
		ax.set_ylabel('True Positive Rate')
		ax.set_xlim(0, 1)
		ax.set_ylim(0, 1)
		
		ax.grid(True, linestyle=':', alpha=0.3)
		ax.legend(loc='lower right', frameon=True, fancybox=True, framealpha=0.9)
		
		plt.title('ROC Curve Comparison')
		plt.tight_layout(pad=1.5)
		
		if save_path:
			plt.savefig(f"{self.save_dir /save_path}.png",
						bbox_inches='tight',
						dpi=300)
		return fig
	
	def plot_precision_recall_curve(self, results: Dict, save_path: Optional[str] = None) -> plt.Figure:
		"""
		Create a publication-ready Precision-Recall curve comparison.
		
		Args:
			results: Dictionary containing results for both models
			save_path: Optional path to save the plot
		"""
		fig, ax = plt.subplots(figsize=(6, 5))
		
		for model_name in ['Model 1', 'Model 2']:
			precision, recall, _ = results[model_name]['test_metrics']['precision_recall_curve']
			auc_pr = results[model_name]['test_metrics']['average_precision']
			
			ax.plot(recall, precision, label=f'{model_name} (AUC = {auc_pr:.3f})',
					color=self.colors[model_name],
					linewidth=1.5)
		
		ax.set_xlabel('Recall')
		ax.set_ylabel('Precision')
		ax.set_xlim(0, 1)
		ax.set_ylim(0, 1)
		
		ax.grid(True, linestyle=':', alpha=0.3)
		ax.legend(loc='lower left', frameon=True, fancybox=True, framealpha=0.9)
		
		plt.title('Precision-Recall Curve Comparison')
		plt.tight_layout(pad=1.5)
		
		if save_path:
			plt.savefig(f"{self.save_dir /save_path}.png",
						bbox_inches='tight',
						dpi=300)


		return fig