import nltk
from typing import Set
from nltk.corpus import stopwords
from nltk.corpus import wordnet

REQUIRED_NLTK_RESOURCES = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']

def setup_nltk() -> None:
	for resource in REQUIRED_NLTK_RESOURCES:
		try:
			nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else resource)
		except LookupError:
			nltk.download(resource, quiet=True)

def get_stopwords() -> Set[str]:
	return set(stopwords.words('english'))

def get_wordnet_pos(treebank_tag):
	if treebank_tag.startswith('J'):
		return wordnet.ADJ
	elif treebank_tag.startswith('V'):
		return wordnet.VERB
	elif treebank_tag.startswith('N'):
		return wordnet.NOUN
	elif treebank_tag.startswith('R'):
		return wordnet.ADV
	else:
		return wordnet.NOUN