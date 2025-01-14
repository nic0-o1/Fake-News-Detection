from .Attention import Attention
from .BaseModule import BaseModule


class LSTMWithAttention(BaseModule):
    def __init__(self, embedding_dim, hidden_dim, output_dim, dropout_rate=0.7):
        super(LSTMWithAttention, self).__init__(embedding_dim, hidden_dim, output_dim, dropout_rate)
        self.attention = Attention(hidden_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        lstm_out = lstm_out.permute(0, 2, 1)
        lstm_out = self.batch_norm(lstm_out).permute(0, 2, 1)
        lstm_out = self.dropout(lstm_out)

        context, attention_weights = self.attention(lstm_out)
        context = context.squeeze(1)
        context = self.dropout(context)
        output = self.fc(context)

        return output, attention_weights