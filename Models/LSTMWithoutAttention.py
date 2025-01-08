from .BaseModule import BaseModule

class LSTMWithoutAttention(BaseModule):
	def __init__(self, embedding_dim, hidden_dim, output_dim, dropout_rate = 0.7):
		super(LSTMWithoutAttention, self).__init__(embedding_dim, hidden_dim, output_dim, dropout_rate)
	
	def forward(self, x):
		lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim)

		# Take the final hidden state (last time step) as the context vector
		final_hidden_state = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
		# Batch Normalisation and Dropout
		final_hidden_state = self.batch_norm(final_hidden_state)
		final_hidden_state = self.dropout(final_hidden_state)
		output = self.fc(final_hidden_state)  # (batch_size, output_dim)

		return output