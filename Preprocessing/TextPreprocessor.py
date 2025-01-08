import re
from typing import Optional, List
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

from .utils import get_stopwords, get_wordnet_pos, setup_nltk
class TextPreprocessor:
	"""
	A class for preprocessing text data with various cleaning and normalization options.
	"""
	
	def __init__(self):
		"""Initialize the TextPreprocessor with required tools."""
		setup_nltk()
		self.lemmatizer = WordNetLemmatizer()
		self.stop_words = get_stopwords()
		self.word_pattern = re.compile(r'[^a-zA-Z\s]')
	
	def clean_text(self, text: str) -> str:
		"""
		Remove special characters and convert text to lowercase.
		
		Args:
			text (str): Input text
			
		Returns:
			str: Cleaned text
		"""
		return self.word_pattern.sub('', text.lower())
	
	def tokenize_and_filter(self, text: str) -> List[str]:
		"""
		Tokenize text and remove stop words.
		
		Args:
			text (str): Input text
			
		Returns:
			List[str]: List of filtered tokens
		"""
		tokens = word_tokenize(text)
		return [token for token in tokens if token not in self.stop_words]
	
	def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
		"""
		Lemmatize a list of tokens.
		
		Args:
			tokens (List[str]): List of tokens
			
		Returns:
			List[str]: List of lemmatized tokens
		"""
		pos_tags = pos_tag(tokens)  # Get POS tags for tokens
		return [
			self.lemmatizer.lemmatize(token, get_wordnet_pos(tag))
			for token, tag in pos_tags
		]
	
	def preprocess_text(self, text: Optional[str]) -> Optional[str]:
		"""
		Preprocess text by cleaning, tokenizing, removing stop words, and lemmatizing.
		
		Args:
			text (Optional[str]): Input text
			
		Returns:
			Optional[str]: Preprocessed text or None if input is invalid
		"""
		if not isinstance(text, str):
			return None
			
		if not text.strip():
			return None
			
		try:
			# Clean text
			cleaned_text = self.clean_text(text)
			
			# Tokenize and filter stop words
			filtered_tokens = self.tokenize_and_filter(cleaned_text)
			
			# Lemmatize tokens
			lemmatized_tokens = self.lemmatize_tokens(filtered_tokens)
			
			# Join tokens back into text
			processed_text = ' '.join(lemmatized_tokens)
			
			return processed_text
			
		except Exception as e:
			return None
