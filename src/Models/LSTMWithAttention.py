from typing import Tuple, Optional
import torch
import torch.nn as nn

from .Attention import Attention
from .BaseModule import BaseModule
class LSTMWithAttention(BaseModule):
    """LSTM model with attention mechanism.
    
    Args:
        embedding_dim (int): Dimension of input embeddings
        hidden_dim (int): Dimension of hidden LSTM state
        output_dim (int): Dimension of output
        dropout_rate (float, optional): Dropout rate. Defaults to 0.7
    """
    def __init__(
        self, 
        embedding_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        dropout_rate: float = 0.7
    ) -> None:
        super().__init__(embedding_dim, hidden_dim, output_dim, dropout_rate)
        self.attention = Attention(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - Output tensor of shape (batch_size, output_dim)
                - Attention weights tensor of shape (batch_size, seq_len)
        """
        lstm_out, _ = self.lstm(x)
        
        # Apply batch normalization
        normalized = self.batch_norm(lstm_out.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Apply dropout
        dropped = self.dropout(normalized)
        
        # Apply attention
        context, attention_weights = self.attention(dropped)
        context = context.squeeze(1)
        
        # Apply dropout to context
        context = self.dropout(context)
        
        # Project to output dimension
        output = self.fc(context)
        
        return output, attention_weights