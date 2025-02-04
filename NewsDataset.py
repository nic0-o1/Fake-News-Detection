import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict

class NewsDataset(Dataset):
	"""
	A PyTorch Dataset for converting news texts to Word2Vec embeddings for LSTM models.
	
	This dataset handles the conversion of text data to fixed-length sequences of word vectors
	using a pre-trained Word2Vec model. It supports padding for shorter sequences and truncation
	for longer ones.
	"""
	
	def __init__(self, texts, labels, word2vec_model, max_len):
		"""
		Initialize the dataset with texts, labels, and Word2Vec model.

		Parameters
		----------
		texts : pandas.Series
			Series containing the news articles or text data.
		labels : pandas.Series
			Series containing the corresponding labels for each text.
		word2vec_model : gensim.models.Word2Vec
			Pre-trained Word2Vec model used for converting words to vectors.
		max_len : int
			Maximum sequence length for padding/truncation.

		Raises
		------
		ValueError
			If texts and labels have different lengths or if max_len is not positive.
		"""
		if len(texts) != len(labels):
			raise ValueError("Length of texts and labels must match")
		if max_len <= 0:
			raise ValueError("max_len must be positive")
			
		self.texts = texts
		self.labels = labels
		self.word2vec_model = word2vec_model
		self.max_len = max_len
		self.vector_size = word2vec_model.vector_size
		self.pad_vector = np.zeros(self.vector_size)

	def __len__(self) -> int:
		"""
		Get the number of samples in the dataset.

		Returns
		-------
		int
			The total number of text samples in the dataset.
		"""
		return len(self.texts)

	def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
		"""
		Get a single sample from the dataset.

		This method converts a text sample into a sequence of word vectors. Words not found
		in the Word2Vec vocabulary are replaced with zero vectors. The sequence is either
		padded or truncated to match max_len.

		Parameters
		----------
		idx : int
			Index of the sample to retrieve.

		Returns
		-------
		dict
			A dictionary containing:
			- 'embeddings': torch.Tensor of shape (max_len, vector_size)
			- 'label': torch.Tensor containing the label
		"""
		text = str(self.texts.iloc[idx])
		label = self.labels.iloc[idx]

		# Get word vectors, using pad vector for unknown words
		vectors = []
		for word in text.split():
			try:
				vectors.append(self.word2vec_model.wv[word])
			except KeyError:
				vectors.append(self.pad_vector)

		# Handle empty texts
		if not vectors:
			vectors = [self.pad_vector] * self.max_len
		
		# Pad or truncate sequence to max_len
		if len(vectors) < self.max_len:
			vectors.extend([self.pad_vector] * (self.max_len - len(vectors)))
		else:
			vectors = vectors[:self.max_len]

		return {
			'embeddings': torch.tensor(np.array(vectors), dtype=torch.float32),
			'label': torch.tensor(label, dtype=torch.float32)
		}