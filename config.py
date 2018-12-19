import os
import re
from multiprocessing import cpu_count
from typing import Dict, Pattern


# allow linking multiple copies of the OpenMP runtime into the program
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# batch size
BATCH_SIZE: int = 16

# word vector size
VEC_SIZE: int = 128

# maximum query word count
MAX_QUERY_WC: int = 40

# maximum doc word count
MAX_DOC_WC: int = 160

# number of bins
BIN_NUM: int = 201

# dropout rate
DROPOUT_RATE: float = 0.5

# initial learning rate
INITIAL_LR: float = 1E-3

# learning rate decay over each update
INITIAL_DECAY: float = 1E-4

# regularization parameter
REGULARIZATION_PARAM: float = 1E-5

# workers
WORKERS: int = cpu_count()
WORKERS = max(1, int(0.75 * WORKERS), WORKERS - 2)

# label_lists
POSITIVE: int = 1
NEGATIVE: int = 0

DATA_DIR: str = os.path.join('.', 'data')
DATA_FILE_PATH: Dict[str, str] = {
    name: os.path.join(DATA_DIR, f'{name}-set.data')
    for name in ('train', 'validation', 'test')
}

# test data not given yet
del DATA_FILE_PATH['test']

FIGURE_DIR: str = os.path.join('.', 'figures')
MODEL_DIR: str = os.path.join('.', 'models')
LOG_DIR: str = os.path.join('.', 'logs')

for DIR in (FIGURE_DIR, MODEL_DIR, LOG_DIR):
    os.makedirs(DIR, exist_ok = True)

del DIR

DICTIONARY_PATH: str = os.path.join(MODEL_DIR, 'dictionary.txt')
WORD2VEC_MODEL_PATH: str = os.path.join(MODEL_DIR, 'word2vec.model')
LATEST_MODEL_PATH: str = os.path.join(MODEL_DIR, 'latest.h5')
LOG_FILE_PATH: str = os.path.join(LOG_DIR, 'logs.csv')

MODEL_FILE_PATTERN: Pattern = re.compile(r'.*epoch(?P<epoch>\d*)_acc(?P<val_acc>[\d.]*)\.h5')
MODEL_FMT_STR: str = os.path.join(MODEL_DIR, 'epoch{epoch:02d}_acc{val_acc:.4f}.h5')

del os, re, cpu_count, Dict, Pattern
