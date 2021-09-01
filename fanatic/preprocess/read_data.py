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

# field definitions for reddit data
REDDIT_DATASET_INPUT_FIELD = "title"
REDDIT_DATASET_LABEL_FIELD = "subreddit"
REDDIT_DATASET_ID_FIELD = "id"


def read_file(
    data_file: str,
    data: Dict,
    n_read: Optional[int] = None,
    min_valid_tokens: int = 3,
    subreddit_labels_list: Optional[List] = None,
) -> int:
    """
    Read a (reddit) zst data file
    Args:
        data_file (string): file path
        data (dict): where the read data is stored
        n_read (int or None): Number of *valid* lines to read (filtered lines don't count). Set to None to read everything
        min_valid_tokens (int): Titles with fewer than this are automatically filtered out
        subreddit_labels_list (list or None): When specified, it is a list of subreddits, and only titles from these
                                          subreddits are considered (everything else is filtered)
    Returns:
        read_lines (int): Number of read lines
        (data) (dict): Although not explicitly returned, this is filled and used downstream
    """

    # init tracking variables
    failed_lines = 0
    read_lines = 0
    duplicate_lines = 0
    ids = set()

    with open(data_file, "rb") as fh:
        # basic code to read in zst files from: https://www.reddit.com/r/pushshift/comments/ajmcc0/information_and_code_examples_on_how_to_use_the/ef012vk/
        dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
        with dctx.stream_reader(fh) as reader:
            previous_line = ""
            while True:
                if n_read is not None and read_lines >= n_read:
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
                            json_line[REDDIT_DATASET_INPUT_FIELD]
                            .encode(encoding="UTF-8", errors="strict")
                            .decode("UTF-8")
                        )
                        label = json_line[REDDIT_DATASET_LABEL_FIELD]
                        id = json_line[REDDIT_DATASET_ID_FIELD]

                        # if provided, filter data *not* in the subreddit_labels_list
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

                        if n_read is not None and read_lines >= n_read:
                            break
                    except:
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
    n_read: Optional[int] = None,
    min_valid_tokens: int = 3,
    subreddit_labels_list: Optional[List] = None,
) -> Dict[str, List]:
    """
    Read a a list of zst files.

    Args:
        data_files (List of strings): list of zst data files to read
        data_type (string): reddit or twitter
        n_read (int or None): Number of *valid* lines to read (filtered lines don't count). Set to None to read everything
        min_valid_tokens (int): Titles with fewer than this are automatically filtered out
        subreddit_labels_list: When specified, it is a dictionary of subreddits and associated coherent/noise label

    Returns:
        data (dict): the read data
    """
    if n_read is None:
        logger.info("n_read = None, reading all documents from all data files")

    data: Dict[str, Any] = {}
    n_read_remaining = n_read  # remaining lines left to read
    for data_file in data_files:
        valid_titles_read = read_file(
            data_file,
            data,
            n_read=n_read_remaining,
            min_valid_tokens=min_valid_tokens,
            subreddit_labels_list=subreddit_labels_list,
        )
        logger.info(f"Completed reading data file: {data_file}")
        if n_read_remaining is not None:
            n_read_remaining -= valid_titles_read
            if n_read_remaining <= 0:
                break
    return data
