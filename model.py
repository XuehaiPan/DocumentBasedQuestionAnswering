import os
from typing import List
from glob import glob
import keras
from config import VEC_SIZE, MAX_QUERY_WC, BIN_NUM, \
    INITIAL_LR, INITIAL_DECAY, REGULARIZATION_PARAM, \
    MODEL_DIR, LATEST_MODEL_PATH, MODEL_FILE_PATTERN, FIGURE_DIR


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
            model_path = model_paths[0]
            print(f'best_model_path = {model_path}')
        except IndexError:
            model_path = LATEST_MODEL_PATH
    try:
        return keras.models.load_model(filepath = model_path)
    except OSError:
        pass
    
    # Inputs
    embedded_query = keras.layers.Input(shape = (MAX_QUERY_WC, VEC_SIZE), name = 'Embedded_query')
    bin_sum = keras.layers.Input(shape = (MAX_QUERY_WC, BIN_NUM), name = 'Bin_sum')
    
    # Hidden Layers
    hidden_layer = bin_sum
    hidden_layer_sizes: List[int] = [128, 64, 32, 16, 1]
    for i, units in enumerate(hidden_layer_sizes, start = 1):
        Dense_i = keras.layers.Dense(units = units, activation = None, use_bias = True,
                                     kernel_regularizer = keras.regularizers.l2(l = REGULARIZATION_PARAM),
                                     name = f'Dense_{i}')
        LeakyReLU_i = keras.layers.LeakyReLU(alpha = 0.1, name = f'LeakyReLU_{i}')
        hidden_layer = Dense_i(hidden_layer)  # shape == [batch_size, MAX_QUERY_WC, units]
        hidden_layer = LeakyReLU_i(hidden_layer)  # shape == [batch_size, MAX_QUERY_WC, units]
    Reshape_5 = keras.layers.Reshape(target_shape = (MAX_QUERY_WC,), name = 'Reshape_5')
    last_hidden_layer = Reshape_5(hidden_layer)  # shape == [batch_size, MAX_QUERY_WC]
    
    # Hidden Layer to Output Layer
    Dense_6 = keras.layers.Dense(units = 1, activation = None, use_bias = False,
                                 kernel_regularizer = keras.regularizers.l2(l = REGULARIZATION_PARAM),
                                 name = 'Dense_6')
    Reshape_6 = keras.layers.Reshape(target_shape = (MAX_QUERY_WC,), name = 'Reshape_6')
    Softmax_6 = keras.layers.Softmax(name = 'Softmax_6')
    query_weights = Dense_6(embedded_query)  # shape == [batch_size, MAX_QUERY_WC, 1]
    query_weights = Reshape_6(query_weights)  # shape == [batch_size, MAX_QUERY_WC]
    query_weights = Softmax_6(query_weights)  # shape == [batch_size, MAX_QUERY_WC]
    
    # Output Layer
    Dot_7 = keras.layers.Dot(axes = [-1, -1], name = 'Dot_7')
    Sigmoid_7 = keras.layers.Activation(activation = 'sigmoid', name = 'Sigmoid_7')
    logits = Dot_7([query_weights, last_hidden_layer])  # shape == [batch_size, 1]
    prediction = Sigmoid_7(logits)  # shape == [batch_size, 1]
    
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
