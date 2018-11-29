from typing import Set, Dict
from Config import DATA_FILE_PATH
from DataSet import quest_ans_label_generator


QUESTION_NUM: Dict[str, int] = {}
for dataset in DATA_FILE_PATH:
    quest_set: Set[str] = set()
    for quest, _, _ in quest_ans_label_generator(dataset = dataset):
        quest_set.add(quest)
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
