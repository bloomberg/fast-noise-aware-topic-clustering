from fanatic.preprocess.nltk_preprocessor import NLTKPreprocessor

DATA_INPUT = [
        {"text": "I ate 10 hotdogs at the baseball game. What about yourself?"},
        {"text": "What is the Meaning of life????"},
    ]
PREPROCESSOR_OUTPUT = [{'text': 'I ate 10 hotdogs at the baseball game. What about yourself?', 'tokens': ['ate', '10', 'hotdogs', 'baseball', 'game'], 'norm_tokens': ['ate', '__NUMBER__', 'hotdogs', 'baseball', 'game']}, {'text': 'What is the Meaning of life????', 'tokens': ['meaning', 'life'], 'norm_tokens': ['meaning', 'life']}]

def test_nltk_preprocessor():
    # GIVEN
    data = DATA_INPUT

    # WHEN
    engine = NLTKPreprocessor()
    preprocessed_data_generator = engine.preprocess(data)
    preprocessed_data = list(preprocessed_data_generator)

    # THEN
    assert preprocessed_data == PREPROCESSOR_OUTPUT


def test_nltk_featurizer():
    pass
