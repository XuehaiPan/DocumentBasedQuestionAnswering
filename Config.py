import os


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# labels
POSITIVE = 1
NEGATIVE = 0

DATA_DIR = './data/'
DATA_FILE_PATH = {
    name: os.path.join(DATA_DIR, '{}-set.data'.format(name))
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
