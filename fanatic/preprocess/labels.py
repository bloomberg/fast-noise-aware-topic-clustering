# Copyright 2021 Bloomberg L.P.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

NOISE_LABEL = "NOISE_LABEL"  # label to assign to noise subreddits
DEFAULT_MAX_N_READ_FOR_NOISE_FILTER = 10000000  # just a very large number


def load_subreddit_labels(subreddit_labels_file: str) -> Dict[str, str]:
    """Load mapping between subreddit and label. Noise subreddits get NOISE_LABEL and otherwise the subreddit is the label.

    Args:
        subreddit_labels_file (str): Filepath of the subreddit_labels file

    Returns:
        subreddit_labels: Contains the subreddit / label mapping.
    """
    # stats
    unique_derived_labels = set()
    n_excluded_annotations = 0

    # fill annotation labels
    with open(subreddit_labels_file) as f:
        subreddit_labels_raw = json.load(f)

    subreddit_labels = {}
    for subreddit, data in subreddit_labels_raw.items():
        derived_label = subreddit if data["coherent_topic"] is True else NOISE_LABEL
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


def prepare_final_dataset_and_labels(
    data: Dict[str, Any], subreddit_labels: Optional[Dict[str, str]] = None
) -> Tuple[List[Dict], Dict[str, str]]:
    """Prepare dataset and labels for clustering. Determine a 'derived' label from the subreddit_labels.

    Args:
        data: The dataset
        subreddit_labels: Contains the subreddit / label mapping when specified

    Returns:
        final_data: the data in proper format for downstream clustering
        final_derived_labels: contains a mapping between datum id and derived label.
    """
    final_derived_labels = {}
    final_data = []
    for subreddit, subreddit_data in data.items():
        for title in subreddit_data:
            # if we have annotations available, use them to get a more informed label,
            # otherwise use the subreddit as the label
            derived_label = (
                subreddit_labels[subreddit]
                if subreddit_labels is not None
                else subreddit
            )

            # prepare title / label
            final_derived_labels[title["id"]] = derived_label
            final_data.append(
                {
                    "text": title["text"],
                    "id": title["id"],
                    "label": derived_label,
                }
            )
    return final_data, final_derived_labels
