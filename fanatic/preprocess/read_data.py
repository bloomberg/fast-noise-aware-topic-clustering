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
import sys
from typing import Any, Dict, List, Optional

import zstandard as zstd

logging_format = (
    "%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)s %(message)s"
)
logging.basicConfig(level=logging.INFO, format=logging_format)
logger = logging.getLogger(__name__)

# dataset field definitions
DATASET_INPUT_FIELD = "title"  # defines the field containing the input text
DATASET_LABEL_FIELD = "subreddit"  # defines the field containing the label
DATASET_ID_FIELD = "id"  # defines the field containing the unique identifier


def read_file(
    data_file: str,
    data: Dict,
    num_docs_read: Optional[int] = None,
    min_valid_tokens: int = 3,
    subreddit_labels_list: Optional[List] = None,
) -> int:
    """Read a zst data file.

    Args:
        data_file: File path to data file
        data: Dict to store data
        num_docs_read: Number of valid lines to read (filtered lines don't count). Set to None to read everything
        min_valid_tokens: minimum number of tokens required to add to the dataset.
        subreddit_labels_list: When specified, restrict the data universe to these labels

    Returns:
        read_lines (int): Number of valid read lines
    """

    # init tracking variables
    failed_lines = 0
    read_lines = 0
    duplicate_lines = 0
    ids = set()

    with open(data_file, "rb") as fh:
        # basic code to read in zst files from:
        # https://www.reddit.com/r/pushshift/comments/ajmcc0/information_and_code_examples_on_how_to_use_the/ef012vk/
        dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
        with dctx.stream_reader(fh) as reader:
            previous_line = ""
            while True:
                if num_docs_read is not None and read_lines >= num_docs_read:
                    break

                chunk = reader.read(2 ** 24)  # 16mb chunks
                if not chunk:
                    break

                string_data = chunk.decode("utf-8")
                lines = string_data.split("\n")
                for i, line in enumerate(lines[:-1]):
                    if i == 0:
                        line = previous_line + line

                    try:
                        json_line = json.loads(line)
                        title = str(
                            json_line[DATASET_INPUT_FIELD]
                            .encode(encoding="UTF-8", errors="strict")
                            .decode("UTF-8")
                        )
                        label = json_line[DATASET_LABEL_FIELD]
                        id = json_line[DATASET_ID_FIELD]

                        # if provided, restrict data universe to labels in subreddit_labels_list
                        if (
                            subreddit_labels_list is not None
                            and label not in subreddit_labels_list
                        ):
                            continue

                        # filter out very short titles
                        if len(title.split()) < min_valid_tokens:
                            continue

                        # filter out duplicate ids
                        if id in ids:
                            duplicate_lines += 1
                            continue

                        # add title to data dict
                        if label not in data:
                            data[label] = []

                        # append title and unique thread id
                        data[label].append({"text": title, "id": id})
                        ids.add(id)
                        read_lines += 1

                        if num_docs_read is not None and read_lines >= num_docs_read:
                            break
                    except Exception:
                        failed_lines += 1

                    if i % 1000 == 0:
                        logger.info(f"Processed {read_lines} lines")
                        sys.stdout.flush()

                previous_line = lines[-1]

    # logging
    logger.info(f"Number of Failed lines: {failed_lines}")
    logger.info(f"Number of Valid lines read: {read_lines}")
    logger.info(f"Number of Duplicate lines: {duplicate_lines}")
    return read_lines


def read_files(
    data_files: List[str],
    num_docs_read: Optional[int] = None,
    min_valid_tokens: int = 3,
    subreddit_labels_list: Optional[List] = None,
) -> Dict[str, List]:
    """Read a a list of zst files.

    Args:
        data_files: List of data file paths
        num_docs_read: Number of valid lines to read (filtered lines don't count). Set to None to read everything
        min_valid_tokens: minimum number of tokens required to add to the dataset.
        subreddit_labels_list: When specified, restrict the data universe to these labels

    Returns:
        data: the read-in data
    """
    data: Dict[str, Any] = {}
    num_docs_read_remaining = num_docs_read  # remaining lines left to read
    for data_file in data_files:
        valid_titles_read = read_file(
            data_file,
            data,
            num_docs_read=num_docs_read_remaining,
            min_valid_tokens=min_valid_tokens,
            subreddit_labels_list=subreddit_labels_list,
        )
        logger.info(f"Completed reading data file: {data_file}")
        if num_docs_read_remaining is not None:
            num_docs_read_remaining -= valid_titles_read
            if num_docs_read_remaining <= 0:
                break
    return data
