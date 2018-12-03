import os
from typing import List
from glob import glob
from tensorflow import keras
from config import MODEL_DIR, LATEST_MODEL_PATH, MODEL_FILE_PATTERN, INITIAL_LR, INITIAL_DECAY


def get_model_paths(sortby: str = 'epoch', reverse: bool = False) -> List[str]:
    assert sortby in ('epoch', 'acc', 'val_acc')
    if 'acc' in sortby:
        sortby = 'val_acc'
    model_paths: List[str] = glob(pathname = os.path.join(MODEL_DIR, 'epoch*_acc*.h5'))
    model_paths.sort(key = lambda file: float(MODEL_FILE_PATTERN.match(file).group(sortby)),
                     reverse = reverse)
    return model_paths


def build_network(model_path: str = None) -> keras.Model:
    if model_path is None:
        try:
            model_paths: List[str] = get_model_paths(sortby = 'val_acc', reverse = True)
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
    inputs = keras.layers.Input(shape = None, name = 'inputs')
    
    # Outputs
    outputs = inputs
    
    model = keras.Model(inputs = inputs, outputs = outputs)
    
    RMSprop_Optimizer = keras.optimizers.RMSprop(lr = INITIAL_LR, decay = INITIAL_DECAY)
    model.compile(optimizer = RMSprop_Optimizer, loss = 'binary_crossentropy', metrics = ['acc'])
    model.save(filepath = LATEST_MODEL_PATH)
    
    return model
