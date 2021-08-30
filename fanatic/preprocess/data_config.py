import argparse

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