from fanatic.preprocess.nltk_preprocessor import NLTKPreprocessor

def test_nltk_preprocessor():
    documents = [
        "I am myself and you are yourself",
        "What is the Meaning of life????",
    ]

    engine = NLTKPreprocessor()
    preprocessed_documents = engine.preprocess(documents)
    assert 'document_tokens' in preprocessed_documents.keys()
    assert preprocessed_documents['document_tokens'] == [[], ['meaning', 'life']]