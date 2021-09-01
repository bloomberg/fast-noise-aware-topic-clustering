"""Functions associated with filtering the data according to a specified noise percentage."""
import logging
import random
from typing import Any, Dict, List, Tuple

from fanatic.preprocess.labels import NOISE_LABEL

logger = logging.getLogger(__name__)


def filter_data_by_noise_percentage(
    data: Dict[str, List],
    n_read: int,
    subreddit_noise_percentage: float,
    subreddit_labels: Dict,
    seed: int,
) -> Dict[str, Any]:
    """
    Filter the dataset to achieve the specified noise percentage. Honor the specified `n_read` value if possible, but the noise percentage is prioritzed.

    Args:
        data: The data
        n_read: Number of documents the final dataset should contain.
        subreddit_noise_percentage: Fraction of noise to read.
        subreddit_labels: Contains the subreddit / label mapping
        seed: Random seed used for shuffling the dataset.

    Returns:
        filtered_data: The filtered data with the appropriate noise percentage and (if possible) number of documents
    """

    # separate coherent and noise data
    (
        coherent_data,
        coherent_subreddit_names,
        noise_data,
        noise_subreddit_names,
    ) = _separate_coherent_and_noise_data(data, subreddit_labels)

    # create and shuffle coherent/noise indices
    random.seed(seed)
    coherent_indices = list(range(len(coherent_data)))
    noise_indices = list(range(len(noise_data)))
    random.shuffle(coherent_indices)
    random.shuffle(noise_indices)

    # calculate number of coherent/noise titles you require based on how coherent many docs were *actually* read
    n_coherent_docs = len(coherent_indices)
    n_required_coherent_titles = int(n_read - subreddit_noise_percentage * n_read)
    n_required_noise_titles = n_read - n_required_coherent_titles
    n_required_noise_titles = min(
        max(n_required_noise_titles, 0), n_read
    )  # avoid potential off-by-1 errors
    if n_required_coherent_titles > n_coherent_docs:
        logger.warning(
            f"Not enough actual coherent documents to fulfill n_read={n_read} and noise percentage={100 * subreddit_noise_percentage}% since "
            f"Required coherent docs = {n_required_coherent_titles}, actual = {n_coherent_docs}. "
            f"Reducing total number of read documents to maintain noise percentage"
        )
        n_required_coherent_titles = n_coherent_docs
        n_required_noise_titles = int(
            n_required_coherent_titles
            * subreddit_noise_percentage
            / (1 - subreddit_noise_percentage)
        )

    # select required number of coherent/noise docs
    coherent_indices = coherent_indices[:n_required_coherent_titles]
    noise_indices = noise_indices[:n_required_noise_titles]

    # get filtered data
    filtered_data: Dict[str, Any] = {}
    counts_by_label: Dict[str, int] = {}
    _add_data_to_filtered_set(
        filtered_data, counts_by_label, coherent_data, coherent_subreddit_names, coherent_indices
    )
    _add_data_to_filtered_set(
        filtered_data, counts_by_label, noise_data, noise_subreddit_names, noise_indices
    )

    # log some stats
    n_coherent_titles = len(coherent_indices)
    n_noise_titles = len(noise_indices)
    actual_noise_percentage = n_noise_titles / (n_noise_titles + n_coherent_titles)
    logger.info(f"noise percentage achieved = {actual_noise_percentage}.")
    logger.info(f"Total number of documents={n_noise_titles + n_coherent_titles}")
    print(f"Breakdown of counts by subreddit:", counts_by_label)
    return filtered_data


def _add_data_to_filtered_set(filtered_data: Dict[str, Any], counts_by_label: Dict[str, int], data: List[Dict[str, Any]], labels: List[str], indices: List[int]):
    """Add data to the filtered set.

    Args:
        filtered_data: The filtered data with the appropriate noise percentage.
        counts_by_label: Tracks the number of counts per label
        data: List of coherent or noise data
        labels: Corresponding labels for `data`
        indices: These are (randomly shuffled) indices to insert into `filtered_data`.

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


def _separate_coherent_and_noise_data(data: Dict[str, List], subreddit_labels: Dict[str, str]) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]], List[str]]:
    """
    Separate the coherent from noise titles

    Args:
        data: The data. Each subreddit key contains a list of titles for that subreddit.
        subreddit_labels: Contains the subreddit / label mapping.

    Returns:
        coherent_data: each item is a subreddit datum from a coherent subreddit
        coherent_subreddit_names: the subreddit corresponding to the item in `coherent_data`
        noise_data: each item is a subreddit datum from a noise subreddit
        noise_subreddit_names: the subreddit corresponding to the item in `noise_data`
    """
    # initialize
    coherent_data = []
    coherent_subreddit_names = []
    noise_data = []
    noise_subreddit_names = []

    # separate coherent from noise titles
    for subreddit, titles in data.items():
        subreddit_names = [subreddit] * len(titles)
        label = subreddit_labels[subreddit]
        if label == NOISE_LABEL:
            noise_data += titles
            noise_subreddit_names += subreddit_names
        else:
            coherent_data += titles
            coherent_subreddit_names += subreddit_names
    return coherent_data, coherent_subreddit_names, noise_data, noise_subreddit_names
