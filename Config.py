import os


# allow linking multiple copies of the OpenMP runtime into the program
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# labels
POSITIVE = 1
NEGATIVE = 0

DATA_DIR = './data/'
DATA_FILE_PATH = {
    name: os.path.join(DATA_DIR, f'{name}-set.data')
    for name in ('train', 'validation', 'test')
}

# test data not given yet
del DATA_FILE_PATH['test']

FIGURE_DIR = './figures/'
MODEL_DIR = './models/'
LOG_DIR = './logs/'

for DIR in (FIGURE_DIR, MODEL_DIR, LOG_DIR):
    if not os.path.exists(DIR):
        os.mkdir(DIR)

from DataSet import quest_ans_label_generator


QUESTION_NUM = {}
for dataset in DATA_FILE_PATH.keys():
    quest_set = set()
    for quest, ans, label in quest_ans_label_generator(dataset = dataset):
        quest_set.add(quest)
    QUESTION_NUM[dataset] = len(quest_set)
