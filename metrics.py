from typing import Set, Dict
from config import DATA_FILE_PATH
from dataset import query_doc_label_generator


QUERY_NUM: Dict[str, int] = {}
for dataset in DATA_FILE_PATH:
    query_set: Set[str] = set(map(lambda data_tuple: data_tuple[0],
                                  query_doc_label_generator(dataset = dataset)))
    QUERY_NUM[dataset] = len(query_set)


def mean_average_precision():
    # TODO: implement mean_average_precision
    raise NotImplementedError


def mean_reciprocal_rank():
    # TODO: implement mean_average_precision
    raise NotImplementedError


# add aliases
MAP = mean_average_precision
MRR = mean_reciprocal_rank
