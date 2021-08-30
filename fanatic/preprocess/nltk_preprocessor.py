from fanatic.preprocess.generic_preprocessor import GenericPreprocessor
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from typing import Dict, List


class NLTKPreprocessor(GenericPreprocessor):
    def __init__(self):
        # https://www.nltk.org/api/nltk.tokenize.html
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stopwords = stopwords.words('english')
        super().__init__()
        
    def preprocess(self, documents: List[List[str]]) -> Dict[str, List[str]]:
        preprocessed_documents = [[tok for tok in self.tokenizer.tokenize(doc.lower()) if tok not in self.stopwords] for doc in documents]
        return {"document_tokens": preprocessed_documents}
    
    def embed(self, preprocessed_documents: Dict[str, List[str]]) -> Dict[str, List[float]]:
        pass