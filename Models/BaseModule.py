import torch
import torch.nn as nn


class BaseModule(nn.Module):
	def __init__(self,  embedding_dim, hidden_dim, output_dim, dropout_rate = 0.7):
		super(BaseModule, self).__init__()
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=False, batch_first=True)

		self.batch_norm = nn.BatchNorm1d(hidden_dim)
		self.dropout = nn.Dropout(dropout_rate)
		self.fc = nn.Linear(hidden_dim, output_dim)
	
	def forward(self, x):
		raise NotImplementedError("Subclasses should implement this method.")