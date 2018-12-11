import os
from typing import List
from glob import glob
import tensorflow as tf
from tensorflow import keras
from .config import VEC_SIZE, MAX_QUERY_WC, MAX_DOC_WC, \
    MODEL_DIR, LATEST_MODEL_PATH, MODEL_FILE_PATTERN, \
    INITIAL_LR, INITIAL_DECAY


tf.set_random_seed(seed = 0)


def get_model_paths(sort_by: str = 'epoch', reverse: bool = False) -> List[str]:
    assert sort_by in ('epoch', 'acc', 'val_acc')
    if 'acc' in sort_by:
        sort_by = 'val_acc'
    model_paths: List[str] = glob(pathname = os.path.join(MODEL_DIR, 'epoch*_acc*.h5'))
    model_paths.sort(key = lambda file: float(MODEL_FILE_PATTERN.match(file).group(sort_by)),
                     reverse = reverse)
    return model_paths


def build_network(model_path: str = None) -> keras.Model:
    if model_path is None:
        try:
            model_paths: List[str] = get_model_paths(sort_by = 'val_acc', reverse = True)
            model_path: str = model_paths[0]
            print(f'best_model_path = {model_path}')
        except IndexError:
            model_path: str = LATEST_MODEL_PATH
    
    try:
        return keras.models.load_model(filepath = model_path)
    except OSError:
        pass
    
    # TODO: implement network
    raise NotImplementedError
    
    # Inputs
    query = keras.layers.Input(shape = (MAX_QUERY_WC, VEC_SIZE), name = 'query')
    doc_list = keras.layers.Input(shape = (MAX_DOC_WC, MAX_BIN_CNT), name = 'doc_list')
    
    # Outputs
    outputs = inputs
    
    model = keras.Model(inputs = [query, doc_list], outputs = outputs)
    
    RMSprop_Optimizer = keras.optimizers.RMSprop(lr = INITIAL_LR, decay = INITIAL_DECAY)
    model.compile(optimizer = RMSprop_Optimizer, loss = 'binary_crossentropy', metrics = ['acc'])
    model.save(filepath = LATEST_MODEL_PATH)
    
    return model
