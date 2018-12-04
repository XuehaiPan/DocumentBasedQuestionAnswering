from typing import Set, Dict
from config import DATA_FILE_PATH
from dataset import quest_ans_label_generator


QUESTION_NUM: Dict[str, int] = {}
for dataset in DATA_FILE_PATH:
    quest_set: Set[str] = set(map(lambda data_tuple: data_tuple[0],
                                  quest_ans_label_generator(dataset = dataset)))
    QUESTION_NUM[dataset] = len(quest_set)


def mean_average_precision():
    # TODO: implement mean_average_precision
    raise NotImplementedError


def mean_reciprocal_rank():
    # TODO: implement mean_average_precision
    raise NotImplementedError


# add aliases
MAP = mean_average_precision
MRR = mean_reciprocal_rank
