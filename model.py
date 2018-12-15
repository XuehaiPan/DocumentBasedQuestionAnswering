import os
from typing import List
from glob import glob
import tensorflow as tf
from tensorflow import keras
from config import VEC_SIZE, MAX_QUERY_WC, BIN_NUM, \
    INITIAL_LR, INITIAL_DECAY, REGULARIZATION_PARAM, \
    MODEL_DIR, LATEST_MODEL_PATH, MODEL_FILE_PATTERN, FIGURE_DIR


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
        keras.models.load_model(filepath = model_path)
    except OSError:
        pass
    
    # Inputs
    embedded_query = keras.layers.Input(shape = (MAX_QUERY_WC, VEC_SIZE), name = 'Embedded_query')
    bin_sum = keras.layers.Input(shape = (MAX_QUERY_WC, BIN_NUM), name = 'Bin_sum')
    
    # Hidden Layers
    hidden_layer = bin_sum
    hidden_layer_sizes: List[int] = [64, 32, 16, 1]
    for i, units in enumerate(hidden_layer_sizes, start = 3):
        Dense_i = keras.layers.Dense(units = units, activation = None, use_bias = True,
                                     kernel_regularizer = keras.regularizers.l2(l = REGULARIZATION_PARAM),
                                     name = f'Dense_{i}')
        BatchNorm_i = keras.layers.BatchNormalization(name = f'BatchNorm_{i}')
        Tanh_i = keras.layers.Activation(activation = 'tanh', name = f'Tanh_{i}')
        hidden_layer = Dense_i(hidden_layer)  # shape == [BATCH_SIZE, MAX_QUERY_WC, units]
        hidden_layer = BatchNorm_i(hidden_layer)  # shape == [BATCH_SIZE, MAX_QUERY_WC, units]
        hidden_layer = Tanh_i(hidden_layer)  # shape == [BATCH_SIZE, MAX_QUERY_WC, units]
    Reshape_6 = keras.layers.Reshape(target_shape = (MAX_QUERY_WC,), name = 'Reshape_6')
    last_hidden_layer = Reshape_6(hidden_layer)  # shape == [BATCH_SIZE, MAX_QUERY_WC]
    
    # Hidden Layer to Output Layer
    Dense_7 = keras.layers.Dense(units = 1, activation = None, use_bias = False,
                                 kernel_regularizer = keras.regularizers.l2(l = REGULARIZATION_PARAM),
                                 name = 'Dense_7')
    Reshape_7 = keras.layers.Reshape(target_shape = (MAX_QUERY_WC,), name = 'Reshape_7')
    Softmax_7 = keras.layers.Softmax(name = 'Softmax_7')
    query_weights = Dense_7(embedded_query)  # shape == [BATCH_SIZE, MAX_QUERY_WC, 1]
    query_weights = Reshape_7(query_weights)  # shape == [BATCH_SIZE, MAX_QUERY_WC]
    query_weights = Softmax_7(query_weights)  # shape == [BATCH_SIZE, MAX_QUERY_WC]
    
    # Output Layer
    Dot_8 = keras.layers.Dot(axes = [-1, -1], name = 'Dot_8')
    Sigmoid_8 = keras.layers.Activation(activation = 'sigmoid', name = 'Sigmoid_8')
    logits = Dot_8([query_weights, last_hidden_layer])  # shape == [BATCH_SIZE, 1]
    prediction = Sigmoid_8(logits)  # shape == [BATCH_SIZE, 1]
    
    model = keras.Model(inputs = [embedded_query, bin_sum], outputs = prediction)
    
    RMSprop_optimizer = keras.optimizers.RMSprop(lr = INITIAL_LR, decay = INITIAL_DECAY)
    model.compile(optimizer = RMSprop_optimizer, loss = 'binary_crossentropy', metrics = ['acc'])
    
    model.save(filepath = LATEST_MODEL_PATH)
    
    return model


def main() -> None:
    model: keras.Model = build_network()
    
    model.summary()
    
    try:
        keras.utils.plot_model(model = model,
                               to_file = os.path.join(FIGURE_DIR, 'model.png'),
                               show_shapes = True,
                               show_layer_names = True,
                               rankdir = 'TB')
    except ImportError:
        pass


if __name__ == '__main__':
    main()
