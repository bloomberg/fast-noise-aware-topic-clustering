import argparse

# nltk stopwords run through the preprocessing
STOPWORDS = ['at', 'as', 'after', 'such', 'too', 'do', 'were', 'not', 'needn', 'your', 'didn', 'does',
             'just', 'have', 'mustn', 'now', 'off', 'him', 'hadn', 'myself', 'this', 'don', 'while',
             'because', 'but', 'more', 'mightn', 'once', 'their', 'during', 'when', 'them', 'hers',
             'my', 'been', 'be', 've', 'own', 'wouldn', 't', 're', 'to', 'could', 'would', 'we', 'each',
             'they', 'between', 's', 'is', 'until', 'himself', 'so', 'then', 'd', 'other', 'about',
             "'re", 'any', "'ll", 'couldn', 'isn', 'over', 'than', 'in', 'must', 'from', "'s", 'themselves',
             'theirs', 'yourself', 'i', 'wo', 'no', 'can', 'are', 'and', 'with', 'might', 'shouldn', 'did',
             'or', 'above', 'her', 'below', 'will', 'all', 'there', 'again', 'against', 'down', 'both',
             'nor', 'having', "'ve", 'has', 'ours', 'weren', 'under', 'most', 'ain', 'should', 'up', 'same',
             'of', 'aren', 'its', 'through', 'was', 'had', 'ma', 'these', 'shan', 'y', 'into', 'that',
             'doing', 'haven', 'an', 'yourselves', 'a', 'sha', 'the', 'some', "n't", 'being', 'o', 'he',
             'won', 'it', 'why', 'am', 'very', 'few', 'before', 'here', 'who', 'further', 'how', 'doesn',
             'hasn', 'his', 'those', 'which', 'where', 'wasn', 'only', 'm', 'our', 'need', 'whom', 'what',
             'you', 'yours', "'d", 'she', 'for', 'herself', 'ourselves', 'me', 'll', 'itself', 'out', 'on',
             'if', 'by', '__NUMBER__', '__PHONE__']

DATASET_INPUT_FIELD = 'text'
DATASET_OUTPUT_FIELD = 'label'
DATASET_ID_FIELD = 'id'

# extendible to other datasets
DATASET_FIELDS = {
    'reddit': {
        DATASET_INPUT_FIELD: 'title',
        DATASET_OUTPUT_FIELD: 'subreddit',
        DATASET_ID_FIELD: 'id'
    },
}


# input data types for argparse
def restricted_int(x):
    if x is None or x == 'None':
        return x
    else:
        try:
            x = int(float(x))
            return x
        except:
            raise argparse.ArgumentTypeError("%r not an integer" % (x,))


def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x