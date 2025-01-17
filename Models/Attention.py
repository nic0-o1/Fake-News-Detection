import torch
import torch.nn as nn

class Attention(nn.Module):
	def __init__(self, hidden_dim):
		super(Attention, self).__init__()
		self.attn = nn.Linear(hidden_dim, hidden_dim)
		self.v = nn.Parameter(torch.rand(hidden_dim)) 

	def forward(self, lstm_outputs):
		scores = torch.tanh(self.attn(lstm_outputs)) 

		scores = torch.matmul(scores, self.v)  

		# Compute attention weights
		weights = torch.softmax(scores, dim=1) 

		# Weighted sum of LSTM outputs
		context = torch.bmm(weights.unsqueeze(1), lstm_outputs).squeeze(1) 
		return context, weights