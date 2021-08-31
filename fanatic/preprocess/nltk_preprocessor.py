from fanatic.preprocess.generic_preprocessor import GenericPreprocessor
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from typing import Any, Dict, Generator, List
import re

NUM_RE = re.compile(r'\d+[A-Za-z]{,2}')

class NLTKPreprocessor(GenericPreprocessor):
    def __init__(self):
        # https://www.nltk.org/api/nltk.tokenize.html
        self.tokenizer = RegexpTokenizer(r"\w+")
        self.stopwords = stopwords.words("english")
        super().__init__()

    def preprocess(self, data: List[Dict[str, Any]]) -> Generator[Dict[str, Any], None, None]:
        for d in data:
            text = d["text"]
            d["tokens"] = [
                tok
                for tok in self.tokenizer.tokenize(text.lower())
                if tok not in self.stopwords
            ]
            d["norm_tokens"] = ['__NUMBER__' if NUM_RE.match(tok) else tok for tok in d["tokens"]]
            yield d

    def embed(self, preprocessed_data_generator: Generator[Dict[str, Any], None, None]) -> Generator[Dict[str, Any], None, None]:
        for d in preprocessed_data_generator:
            d["embedding"] = None
            yield d
        

    def featurize(self, data: List[Dict[str, Any]]) -> Generator[Dict[str, Any], None, None]:
        """Combination of preprocess and embed."""
        preprocessed_data_generator = self.preprocess(data)
        embedding_data_generator = self.embed(preprocessed_data_generator)
        return embedding_data_generator
