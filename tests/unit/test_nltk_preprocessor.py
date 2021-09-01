from fanatic.preprocess.nltk_preprocessor import NLTKPreprocessor
import pytest

EMBEDDING_MODEL_FILE = "tests/unit/small_w2v_file.txt"
DATA_INPUT = [
        {"text": "Looking for tattoo design help?"},
        {"text": "My 5 new ankle tats - at idol time tattoo in tulsa, ok"},
    ]
PREPROCESSOR_EXPECTED_OUTPUT = [{'text': 'Looking for tattoo design help?', 'tokens': ['looking', 'for', 'tattoo', 'design', 'help'], 'norm_tokens': ['looking', 'tattoo', 'design', 'help']}, {'text': 'My 5 new ankle tats - at idol time tattoo in tulsa, ok', 'tokens': ['my', '5', 'new', 'ankle', 'tats', 'at', 'idol', 'time', 'tattoo', 'in', 'tulsa', 'ok'], 'norm_tokens': ['__NUMBER__', 'new', 'ankle', 'tats', 'idol', 'time', 'tattoo', 'tulsa', 'ok']}]
FEATURIZED_EXPECTED_CLUSTERING_TOKENS = [{'clustering_tokens': ['looking', 'tattoo', 'design', 'help']}, {'clustering_tokens': ['__NUMBER__', 'new', 'tats', 'time', 'tattoo', 'ok']}]

def test_nltk_preprocessor():
    # GIVEN
    data = DATA_INPUT
    engine = NLTKPreprocessor()

    # WHEN
    preprocessed_data_generator = engine.preprocess(data)
    preprocessed_data = list(preprocessed_data_generator)

    # THEN
    assert preprocessed_data == PREPROCESSOR_EXPECTED_OUTPUT


def test_nltk_featurizer():
    # GIVEN
    data = DATA_INPUT
    engine = NLTKPreprocessor(embedding_model_file=EMBEDDING_MODEL_FILE)

    # WHEN
    featurized_data_generator = engine.featurize(data)
    featurized_data = list(featurized_data_generator)

    # THEN
    assert len(featurized_data) == 2
    for i, datum in enumerate(featurized_data):
        assert datum["clustering_tokens"] == FEATURIZED_EXPECTED_CLUSTERING_TOKENS[i]["clustering_tokens"]
        assert "embedding" in datum.keys()


def test_nltk_fail_featurize():
    # GIVEN
    data = DATA_INPUT
    engine = NLTKPreprocessor()

    # WHEN / THEN
    with pytest.raises(ValueError):
        engine.featurize(data)
