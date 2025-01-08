import torch
import torch.nn as nn

class Attention(nn.Module):
	def __init__(self, hidden_dim):
		super(Attention, self).__init__()
		self.attn = nn.Linear(hidden_dim, hidden_dim)
		self.v = nn.Parameter(torch.rand(hidden_dim))  # Learnable vector

	def forward(self, lstm_outputs):
		# lstm_outputs: (batch_size, seq_len, hidden_dim)
		scores = torch.tanh(self.attn(lstm_outputs))  # (batch_size, seq_len, hidden_dim)

		# Compute attention scores by taking the dot product with self.v
		scores = torch.matmul(scores, self.v)  # (batch_size, seq_len)

		# Compute attention weights
		weights = torch.softmax(scores, dim=1)  # (batch_size, seq_len)

		# Weighted sum of LSTM outputs
		context = torch.bmm(weights.unsqueeze(1), lstm_outputs).squeeze(1)  # (batch_size, hidden_dim)
		return context, weights