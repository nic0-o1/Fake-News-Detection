from .BaseModule import BaseModule
class LSTMWithoutAttention(BaseModule):
	def __init__(self, embedding_dim, hidden_dim, output_dim, dropout_rate):
		super(LSTMWithoutAttention, self).__init__(embedding_dim, hidden_dim, output_dim, dropout_rate)

	def forward(self, x):
		lstm_out, _ = self.lstm(x)
		lstm_out = lstm_out.permute(0, 2, 1) 
		lstm_out = self.batch_norm(lstm_out)
		lstm_out = lstm_out.permute(0, 2, 1)  
		lstm_out = self.dropout(lstm_out)
		final_hidden_state = lstm_out[:, -1, :] 
		output = self.fc(final_hidden_state)  
		return output