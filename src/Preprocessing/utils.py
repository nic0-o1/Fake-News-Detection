from typing import Set, Literal, Optional
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tag import mapping

# Constants
REQUIRED_NLTK_RESOURCES = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']


def setup_nltk() -> None:
	"""
	Downloads required NLTK resources if they are not already present.
	
	This function checks for the existence of required NLTK resources and
	downloads any missing ones silently.
	
	Raises:
		nltk.exceptions.ConnectionError: If download fails due to network issues
	"""
	for resource in REQUIRED_NLTK_RESOURCES:
		try:
			resource_path = f'tokenizers/{resource}' if resource == 'punkt' else resource
			nltk.data.find(resource_path)
		except LookupError:
			nltk.download(resource, quiet=True)

def get_stopwords() -> Set[str]:
	"""
	Retrieves the set of English stopwords from NLTK.
	
	Returns:
		Set[str]: A set containing all English stopwords
	
	Raises:
		LookupError: If stopwords resource is not available
	"""
	return set(stopwords.words('english'))

def get_wordnet_pos(treebank_tag: str) -> str:
	"""
	Maps Penn Treebank POS tags to WordNet POS tags.
	
	Args:
		treebank_tag (str): Penn Treebank POS tag
		
	Returns:
		str: Corresponding WordNet POS tag ('a' for adjective, 'v' for verb,
			 'n' for noun, 'r' for adverb)
		
	Note:
		Default return is 'n' (noun) if no mapping is found
	"""
	if treebank_tag.startswith('J'):
		return 'a'  # adjective
	elif treebank_tag.startswith('V'):
		return 'v'  # verb
	elif treebank_tag.startswith('N'):
		return 'n'  # noun
	elif treebank_tag.startswith('R'):
		return 'r'  # adverb
	else:
		return 'n'  # noun default