import json
import logging

logger = logging.getLogger(__name__)

NOISE_LABEL = "NOISE_LABEL"  # label to assign to noise subreddits
VALID_SUBREDDIT_ANNOTATION_VALUE = True
NOISE_SUBREDDIT_ANNOTATION_VALUE = False
DEFAULT_MAX_N_READ_FOR_NOISE_FILTER = 10000000  # just a very large number


def load_subreddit_labels(subreddit_labels_file):
    """
    load mapping between subreddit and label. 'noise' subreddits get a NOISE_LABEL, and 'valid'
    subreddits get the appropriate label (see get_derived_clustering_label() function for details)

    Args:
        subreddit_labels_file (str): filepath of the subreddit_labels file

    Returns:
        subreddit_labels (dict): (key, value) -> (subreddit, derived_label). See `get_derived_clustering_label()` for
                                  more info on `derived_label`.
    """
    # stats
    unique_derived_labels = set()
    n_excluded_annotations = 0

    # fill annotation labels
    with open(subreddit_labels_file) as f:
        subreddit_labels_raw = json.load(f)

    subreddit_labels = {}
    for subreddit, data in subreddit_labels_raw.items():
        label = data["coherent_topic"]
        derived_label = _get_derived_clustering_label(subreddit, label)
        subreddit_labels[subreddit] = derived_label

        # stats
        unique_derived_labels.add(derived_label)

    # print some stuff
    logger.info(
        f"Excluded {n_excluded_annotations} annotation categories. "
        f"Using {len(subreddit_labels)} different subreddits"
    )
    logger.info(f"{len(unique_derived_labels)} unique derived_labels")
    logger.info("derived_label categories:")
    logger.info(unique_derived_labels)
    return subreddit_labels


def _get_derived_clustering_label(subreddit, label):
    """
    This uses annotation labels (yes/no) to get a `derived_label` - if annotation is 'yes' then use the subreddit
    as the label. If annotation is 'no' then it gets a NOISE_LABEL label.

    Args:
        subreddit (string): The subreddit
        label (string): Whether the subreddit was annotated as a coherent topic (yes/no)

    Returns:
        label (str or None): The appropriate clustering label associated with the subreddit
    """
    if label == VALID_SUBREDDIT_ANNOTATION_VALUE:
        # for positively annotated subreddits, the subreddit is the label
        return subreddit
    elif label == NOISE_SUBREDDIT_ANNOTATION_VALUE:
        # these are subreddits annotated as "no, this subreddit does not represent a coherent topic"
        # all `noise` titles should ideally end up in a single, noise cluster as assigned by the algorithm
        return NOISE_LABEL
    else:
        raise ValueError(f"label={label} is invalid. ")


# TODO: This may not be needed...
def prepare_titles_and_labels(data, subreddit_labels):
    """
    prepare titles and labels for clustering. Specifically use label to determine a 'derived' label
    - i.e. if it was a valid cluster, the label = subreddit, else label = NOISE_LABEL

    Args:
        data (dict): The read data
        subreddit_labels (dict or None): key/value -> (subreddit, label). See get_label() in
                                          annotations.py to see how `label` is assigned

    Returns:
        all_titles (List of dicts): the data in proper format for downstream clustering
        all_derived_labels (dict): key/value -> (id, derived_label)
    """
    all_derived_labels = {}
    all_titles = []
    for subreddit, titles in data.items():
        for title in titles:
            # if we have annotations available, use them to get a more informed label,
            # otherwise use the subreddit as the label
            derived_label = (
                subreddit_labels[subreddit]
                if subreddit_labels is not None
                else subreddit
            )

            # prepare title / label
            all_derived_labels[title["id"]] = derived_label
            all_titles.append(
                {
                    "text": title["text"],
                    "id": title["id"],
                    "label": derived_label,
                    "subreddit": subreddit,
                }
            )
    return all_titles, all_derived_labels
