import logging
import random
from typing import Any, Dict, List

from fanatic.preprocess.labels import NOISE_LABEL

logger = logging.getLogger(__name__)


def filter_data_by_noise_percentage(
    data: Dict[str, List],
    n_read: int,
    subreddit_noise_percentage: float,
    subreddit_labels: Dict,
    seed: int,
) -> Dict[str, List]:
    """
    This filters data to achieve a specified 'noise ratio'. From our annotation task, we determined that some
    subreddits are 'valid' (they represent a coherent topic) and some are 'noise' (they do not represent a coherent
    topic). This function allows the data to be filtered such that exactly `n_read` number of documents are read with
    a `subreddit_noise_percentage` percent of them being noise subreddits

    Args:
        data: the data
        n_read: how much data to read
        subreddit_noise_percentage: Fraction of noise to read. Must be between 0-1
        subreddit_labels: key/value -> (subreddit, annotation=(yes, no)). Determines which subreddits are valid/noise
        seed: random seed for shuffling valid/noise docs

    Returns:
        filtered_data (dict): The filtered data with the appropriate noise percentage
    """

    # ensure noise percentage in [0,1]
    if subreddit_noise_percentage > 1 or subreddit_noise_percentage < 0:
        logger.info(
            f"subreddit_noise_percentage={subreddit_noise_percentage}, limiting to range=[0,1]"
        )
        subreddit_noise_percentage = max(min(subreddit_noise_percentage, 1), 0)

    # separate valid and noise data
    (
        valid_data,
        valid_subreddit_names,
        noise_data,
        noise_subreddit_names,
    ) = _separate_valid_and_noise_data(data, subreddit_labels)

    # create and shuffle valid/noise indices
    random.seed(seed)
    valid_indices = list(range(len(valid_data)))
    noise_indices = list(range(len(noise_data)))
    random.shuffle(valid_indices)
    random.shuffle(noise_indices)

    # calculate number of valid/noise titles you require based on how valid many docs were *actually* read
    n_valid_docs = len(valid_indices)
    n_required_valid_titles = int(n_read - subreddit_noise_percentage * n_read)
    n_required_noise_titles = n_read - n_required_valid_titles
    n_required_noise_titles = min(
        max(n_required_noise_titles, 0), n_read
    )  # avoid potential off-by-1 errors
    if n_required_valid_titles > n_valid_docs:
        logger.warning(
            f"Not enough actual valid documents to fulfill n_read={n_read} and noise percentage={100 * subreddit_noise_percentage}% since "
            f"Required valid docs = {n_required_valid_titles}, actual = {n_valid_docs}. "
            f"Reducing total number of read documents to maintain noise percentage"
        )
        n_required_valid_titles = n_valid_docs
        n_required_noise_titles = int(
            n_required_valid_titles
            * subreddit_noise_percentage
            / (1 - subreddit_noise_percentage)
        )

    # select required number of valid/noise docs
    valid_indices = valid_indices[:n_required_valid_titles]
    noise_indices = noise_indices[:n_required_noise_titles]

    # get filtered data
    filtered_data: Dict[str, Any] = {}
    counts_by_label: Dict[str, int] = {}
    _add_data_to_filtered_set(
        filtered_data, counts_by_label, valid_data, valid_subreddit_names, valid_indices
    )
    _add_data_to_filtered_set(
        filtered_data, counts_by_label, noise_data, noise_subreddit_names, noise_indices
    )

    # log some stats
    n_valid_titles = len(valid_indices)
    n_noise_titles = len(noise_indices)
    actual_noise_percentage = n_noise_titles / (n_noise_titles + n_valid_titles)
    logger.info(f"noise percentage achieved = {actual_noise_percentage}.")
    logger.info(f"Total number of documents={n_noise_titles + n_valid_titles}")
    print(f"Breakdown of counts by subreddit:", counts_by_label)
    return filtered_data


def _add_data_to_filtered_set(filtered_data, counts_by_label, data, labels, indices):
    """
    Add data to the filtered set

    Args:
        filtered_data (dict of lists): The filtered set. (key, value) -> (subreddit, list of titles)
        counts_by_label (dict): (key, value) -> (subreddit, number of titles in `filtered_data` for this subreddit)
        data (list of titles): These are the valid or noise titles to be added to the filtered_data
        labels (list of str): These are the corresponding labels for `data`
        indices (list of ints): These are (randomly shuffled) indices to insert into `filtered_data`

    Returns:
        (nothing officially, but filtered_data is filled)
    """
    for index in indices:
        label = labels[index]
        datum = data[index]
        if label not in filtered_data.keys():
            filtered_data[label] = [datum]
            counts_by_label[label] = 1
        else:
            filtered_data[label].append(datum)
            counts_by_label[label] += 1


def _separate_valid_and_noise_data(data, subreddit_labels):
    """
    Separate the valid from noise titles

    Args:
        data (dict of dicts): (key, value) -> (subreddit, dict of titles for that subreddit)
        subreddit_labels (dict): (key, value) -> (subreddit, label).
                                  `label` is either the subreddit name or NOISE_LABEL

    Returns:
        valid_data (list of dicts): each item is a subreddit title from a valid subreddit
        valid_subreddit_names (list of str): the subreddit corresponding to the item in `valid_data`
        noise_data (list of dicts): each item is a subreddit title from a noise subreddit
        noise_subreddit_names (list of str): the subreddit corresponding to the item in `noise_data`
    """
    # initialize
    valid_data = []
    valid_subreddit_names = []
    noise_data = []
    noise_subreddit_names = []

    # separate valid from noise titles
    for subreddit, titles in data.items():
        subreddit_names = [subreddit] * len(titles)
        label = subreddit_labels[subreddit]
        if label == NOISE_LABEL:
            noise_data += titles
            noise_subreddit_names += subreddit_names
        else:
            valid_data += titles
            valid_subreddit_names += subreddit_names
    return valid_data, valid_subreddit_names, noise_data, noise_subreddit_names
