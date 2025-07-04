import stanza
import spacy
import warnings
warnings.filterwarnings('ignore')


class BaseFeatureExtractor:
    """Base class for all feature extractors"""
    
    def __init__(self):
        self.nlp_stanza = None
        self.nlp_spacy = None
        self._initialized = False
    
    def initialize_nlp(self):
        """Initialize NLP models (lazy loading)"""
        if not self._initialized:
            self.nlp_stanza = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse')
            self.nlp_spacy = spacy.load('en_core_web_sm')
            self._initialized = True
    
    def extract(self, text):
        """Extract features from text. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement extract method")