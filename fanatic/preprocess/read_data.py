import json
from fanatic.preprocess.data_config import DATASET_FIELDS, DATASET_INPUT_FIELD, DATASET_OUTPUT_FIELD, DATASET_ID_FIELD
import sys
from typing import Dict, List, Optional
import zstandard as zstd

import logging
logging_format = '%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)s %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)
logger = logging.getLogger(__name__)


def read_file(data_file: str, data: Dict, data_type: str="reddit", n_read: Optional[int]=None, min_sentence_length: int=3, label_filter_list: Optional[List]=None):
    '''
    Read a (reddit) zst data file
    Args:
        data_file (string): file path
        data (dict): where the read data is stored
        data_type (string): reddit or twitter
        n_read (int or None): Number of *valid* lines to read (filtered lines don't count). Set to None to read everything
        min_sentence_length (int): Titles with fewer than this are automatically filtered out
        label_filter_list (list or None): When specified, it is a list of subreddits, and only titles from these
                                          subreddits are considered (everything else is filtered)
    Returns:
        read_lines (int): Number of read lines
        (data) (dict): Although not explicitly returned, this is filled and used downstream
    '''

    # define variables
    failed_lines = 0
    read_lines = 0
    duplicate_lines = 0
    ids = set()
    input_field = DATASET_FIELDS[data_type][DATASET_INPUT_FIELD]
    output_field = DATASET_FIELDS[data_type][DATASET_OUTPUT_FIELD]
    id_field = DATASET_FIELDS[data_type][DATASET_ID_FIELD]

    with open(data_file, 'rb') as fh:
        # basic code to read in zst files from: https://www.reddit.com/r/pushshift/comments/ajmcc0/information_and_code_examples_on_how_to_use_the/ef012vk/
        dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
        with dctx.stream_reader(fh) as reader:
            previous_line = ""
            while True:
                chunk = reader.read(2**24)  # 16mb chunks
                if not chunk:
                    break

                string_data = chunk.decode('utf-8')
                lines = string_data.split("\n")
                for i, line in enumerate(lines[:-1]):
                    if i == 0:
                        line = previous_line + line
                    data_object = json.loads(line)

                    try:
                        jayson = json.loads(line)
                        title = str(jayson[input_field].encode(encoding='UTF-8', errors='strict').decode('UTF-8'))
                        label = jayson[output_field]
                        id = jayson[id_field]

                        # filter out labels not in the label_filter_list - used mostly for analysis scripts
                        if label_filter_list is not None and label not in label_filter_list:
                            continue

                        # filter out very short titles
                        if len(title.split()) < min_sentence_length:
                            continue

                        # filter out duplicate ids
                        if id in ids:
                            duplicate_lines += 1
                            continue

                        # add title
                        if label not in data:
                            data[label] = []

                        # append title and unique thread id
                        data[label].append({'text': title, 'id': id})
                        ids.add(id)
                        read_lines += 1

                        if n_read is not None and read_lines >= n_read:
                            break
                    except:
                        failed_lines += 1

                    if i % 10000 == 0:
                        logger.info(f"Processed {i} lines")
                        sys.stdout.flush()

                previous_line = lines[-1]

    # logging
    logger.info(f"Number of Failed lines: {failed_lines}")
    logger.info(f"Number of Valid lines read: {read_lines}")
    logger.info(f"Number of Duplicate lines: {duplicate_lines}")
    return read_lines


def read_files(data_files: List[str], data_type: str="reddit", n_read: Optional[int]=None, min_sentence_length: int=3, label_filter_list: Optional[List]=None):
    '''
    Read a a list of zst files.

    Args:
        data_files (List of strings): list of zst data files to read
        data_type (string): reddit or twitter
        n_read (int or None): Number of *valid* lines to read (filtered lines don't count). Set to None to read everything
        min_sentence_length (int): Titles with fewer than this are automatically filtered out
        label_filter_list (list): When specified, it is a list of subreddits, and only consider titles from these subreddits

    Returns:
        data (dict): the read data
    '''
    if n_read is None:
        logger.info("n_read = None, reading all documents from all data files")

    data = {}
    n_read_remaining = n_read   # remaining lines left to read
    for data_file in data_files:
        valid_titles_read = read_file(data_file, data, data_type, n_read=n_read_remaining,
                                      filter_profanity_flag=filter_profanity_flag,
                                      label_filter_list=label_filter_list,
                                      min_sentence_length=min_sentence_length)
        logger.info(f"Completed reading data file: {data_file}")
        if n_read_remaining is not None:
            n_read_remaining -= valid_titles_read
            if n_read_remaining <= 0:
                break
    return data