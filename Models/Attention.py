import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class Attention(nn.Module):
	"""
	Additive attention mechanism (Bahdanau attention).
	
	This attention mechanism computes a weighted sum of LSTM outputs based on learned
	attention weights. It first projects the inputs through a linear layer, applies
	tanh activation, then computes alignment scores using a learned vector.
	
	Args:
		hidden_dim (int): Dimension of the hidden states
	
	Shape:
		- Input: (batch_size, seq_length, hidden_dim)
		- Output: (batch_size, hidden_dim), (batch_size, seq_length)
	"""
	def __init__(self, hidden_dim: int) -> None:
		super().__init__()
		self.hidden_dim = hidden_dim
		
		# Project input hidden states
		self.projection = nn.Linear(hidden_dim, hidden_dim)
		
		# Learnable vector for computing attention scores
		# Initialize using uniform distribution in [-1/sqrt(hidden_dim), 1/sqrt(hidden_dim)]
		self.v = nn.Parameter(
			torch.empty(hidden_dim).uniform_(-1.0 / (hidden_dim ** 0.5), 
										   1.0 / (hidden_dim ** 0.5))
		)

	def forward(self, lstm_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Compute attention weighted context vector and attention weights.
		
		Args:
			lstm_outputs: Hidden states from LSTM (batch_size, seq_length, hidden_dim)
			
		Returns:
			context: Attention weighted sum of hidden states (batch_size, hidden_dim)
			weights: Attention weights (batch_size, seq_length)
		"""
		# Project and apply tanh activation (batch_size, seq_length, hidden_dim)
		scores = torch.tanh(self.projection(lstm_outputs))
		
		# Compute alignment scores (batch_size, seq_length)
		scores = torch.matmul(scores, self.v)
		
		# Normalize attention weights with softmax
		weights = F.softmax(scores, dim=1)
		
		# Compute weighted sum of LSTM outputs
		# weights: (batch_size, 1, seq_length)
		# lstm_outputs: (batch_size, seq_length, hidden_dim)
		# context: (batch_size, 1, hidden_dim) -> (batch_size, hidden_dim)
		context = torch.bmm(weights.unsqueeze(1), lstm_outputs).squeeze(1)
		
		return context, weights