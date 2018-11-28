import os
from typing import Set, Dict


# allow linking multiple copies of the OpenMP runtime into the program
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# labels
POSITIVE: int = 1
NEGATIVE: int = 0

DATA_DIR: str = './data/'
DATA_FILE_PATH: Dict[str, str] = {
    name: os.path.join(DATA_DIR, f'{name}-set.data')
    for name in ('train', 'validation', 'test')
}

# test data not given yet
del DATA_FILE_PATH['test']

FIGURE_DIR: str = './figures/'
MODEL_DIR: str = './models/'
LOG_DIR: str = './logs/'

for DIR in (FIGURE_DIR, MODEL_DIR, LOG_DIR):
    if not os.path.exists(DIR):
        os.mkdir(DIR)

from DataSet import quest_ans_label_generator


QUESTION_NUM: Dict[str, int] = {}
for dataset in DATA_FILE_PATH.keys():
    quest_set: Set[str] = set()
    for quest, ans, label in quest_ans_label_generator(dataset = dataset):
        quest_set.add(quest)
    QUESTION_NUM[dataset] = len(quest_set)
