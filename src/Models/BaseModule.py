from typing import Tuple, Optional
import torch
import torch.nn as nn

class BaseModule(nn.Module):
    """Base LSTM module that serves as a foundation for derived models.
    
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
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            bidirectional=False,
            batch_first=True
        )
        
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim)
            
        Raises:
            NotImplementedError: This method should be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement forward method")