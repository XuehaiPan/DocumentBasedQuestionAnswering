import os
import re
from typing import Dict, Pattern


# allow linking multiple copies of the OpenMP runtime into the program
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# batch size
BATCH_SIZE = 64

# initial learning rate
lr = 1E-3

# learning rate decay over each update
decay = 1E-4

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

LATEST_MODEL_PATH: str = os.path.join(MODEL_DIR, 'latest.h5')
LOG_FILE_PATH: str = os.path.join(LOG_DIR, 'log.csv')

MODEL_FILE_PATTERN: Pattern = re.compile(r'.*epoch(?P<epoch>\d*)_acc(?P<val_acc>[\d.]*)\.h5')
MODEL_FMT_STR: str = os.path.join(MODEL_DIR, 'epoch{epoch:02d}_acc{val_acc:.4f}.h5')
