import argparse
from gensim.models import Word2Vec
from fanatic.preprocess import read_data, nltk_preprocessor, labels
from typing import List

import logging
logging_format = '%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)s %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)
logger = logging.getLogger(__name__)


def train_w2v(token_list: List[str], args: argparse.Namespace) -> None:
    print(f"Training Word2Vec Model")
    model = Word2Vec(token_list, size=args.size, window=args.window, sg=SG, seed=args.seed)
    model.wv.save_word2vec_format(args.output_file, binary=False)
    print("Success!")


def prepare_tokens_list(args: argparse.Namespace) -> List[str]:
    # retrieve annotation subreddits / labels
    subreddit_labels_list = None
    if args.subreddit_labels_file is not None:
        # restrict universe of reddit data to the list of subreddits specified in args.subreddit_labels_file
        subreddit_labels = labels.load_subreddit_labels(args.subreddit_labels_file)
        subreddit_labels_list = list(subreddit_labels.keys())

    data = read_data.read_files(args.data_files, 
                                n_read=args.n_read,
                                min_valid_tokens=args.min_valid_tokens,
                                subreddit_labels_list=subreddit_labels_list)

    # flatten data
    data = [
        datum for _, subreddit_data in data.items() for datum in subreddit_data
    ]

    # preprocess and prepare data
    engine = nltk_preprocessor.NLTKPreprocessor()
    preprocessed_data_generator = engine.preprocess(data)
    token_list = [datum['norm_tokens'] for datum in preprocessed_data_generator]

    logger.info(f"prepared {len(token_list)} documents")
    logger.info(f"first 10 documents sample: {token_list[:10]}")
    return token_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train W2V model')
    # data parameters
    parser.add_argument('--data-files', type=str,
                        nargs='+',
                        default=[
                            "data/RS_2017-11.zst",
                        ],
                        help='data files to use for training w2v')
    parser.add_argument('--n-read', type=int,
                        default=None,  # reads everything
                        help='Number of documents to read. Set to None to read everything from the provided --data-files')
    parser.add_argument('--min-valid-tokens', type=int,
                        default=3,
                        help='Minimum number of words in a sentence for the sentence to be used in clustering; \
                        the first sentence in an inquiry with at least min_valid_tokens words will be used.')
    parser.add_argument('--output-file', type=str,
                        default='w2v_reddit_s300_w5_sg1_RS_2017-11.txt')

    # restrict embedding data universe to the categories of interest
    parser.add_argument('--subreddit-labels-file', type=str,
                        default=f'subreddit_labels.json',
                        help='location of file containing annotation labels')

    # w2v parameters
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sg', type=int, default=1, help='w2v skip-gram')
    parser.add_argument('--size', type=int, default=300, help='w2v size')
    parser.add_argument('--window', type=int, default=5, help='w2v window')
    args = parser.parse_args()
    logger.info('Command line args: {}'.format(args))

    # prepare data
    token_list = prepare_tokens_list(args)

    # train model
    train_w2v(token_list, args)
