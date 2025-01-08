import torch

EMBEDDING_DIM = 300
EARLY_STOP_PATIENCE = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'