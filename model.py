import os
from typing import List
from glob import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from config import VEC_SIZE, MAX_QUERY_WC, MAX_DOC_WC, BIN_NUM, BATCH_SIZE, \
    MODEL_DIR, LATEST_MODEL_PATH, MODEL_FILE_PATTERN, \
    INITIAL_LR, INITIAL_DECAY
from dataset import cut_sentence
from word2vec import get_vectors


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
    def sent2vec(sent, max_wc):
        embedded = np.zeros(shape = (BATCH_SIZE, MAX_QUERY_WC, VEC_SIZE), dtype = np.float32)
        for s in range(BATCH_SIZE):
            word_list = cut_sentence(sentence = sent[s])
            vec_list = get_vectors(word_list = word_list)
            for i, vec in zip(range(max_wc), vec_list):
                embedded[i] = vec
        return embedded
    
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
    
    # Inputs
    embedded_query = keras.layers.Input(shape = (MAX_QUERY_WC, VEC_SIZE), name = 'Embedded_query')
    embedded_doc = keras.layers.Input(shape = (MAX_DOC_WC, VEC_SIZE), name = 'Embedded_doc')
    
    # Calculate QA Matching Matrix
    Dot_1 = keras.layers.Dot(axes = [-1, -1], normalize = True, name = 'Dot_1')
    qa_mat = Dot_1([embedded_query, embedded_doc])  # shape == [BATCH_SIZE, MAX_QUERY_WC, MAX_DOC_WC]
    
    # Input Layer to Hidden Layer
    def cal_bin_sum(input):
        output = np.zeros(shape = (BATCH_SIZE, MAX_QUERY_WC, BIN_NUM))
        for s in range(BATCH_SIZE):
            for i in range(MAX_QUERY_WC):
                for j in range(MAX_DOC_WC):
                    val = input[s, i, j]
                    k = (val + 1) * (BIN_NUM - 1) / 2
                    k = keras.backend.round(k + 0.5)  # k = floor(k)
                    output[s, i, k] += val
        return keras.backend.variable(output)
    
    CalBinSum_2 = keras.layers.Lambda(function = cal_bin_sum, output_shape = (MAX_QUERY_WC, BIN_NUM),
                                      name = 'CalBinSum_2')
    bin_sum = CalBinSum_2(qa_mat)  # shape == [BATCH_SIZE, MAX_QUERY_WC, BIN_NUM]
    
    # Hidden Layers
    hidden_layer = bin_sum
    hidden_layer_sizes = [64, 32, 16, 1]
    for i, units in enumerate(hidden_layer_sizes, start = 3):
        Dense_i = keras.layers.Dense(units = units, activation = None, name = f'Dense_{i}')
        BatchNorm_i = keras.layers.BatchNormalization(name = f'BatchNorm_{i}')
        Tanh_i = keras.layers.Activation(activation = 'tanh', name = f'Tanh_{i}')
        hidden_layer = Dense_i(hidden_layer)  # shape == [BATCH_SIZE, MAX_QUERY_WC, units]
        hidden_layer = BatchNorm_i(hidden_layer)  # shape == [BATCH_SIZE, MAX_QUERY_WC, units]
        hidden_layer = Tanh_i(hidden_layer)  # shape == [BATCH_SIZE, MAX_QUERY_WC, units]
    Reshape_6 = keras.layers.Reshape(target_shape = (MAX_QUERY_WC,), name = 'Reshape_6')
    last_hidden_layer = Reshape_6(hidden_layer)  # shape == [BATCH_SIZE, MAX_QUERY_WC]
    
    # Hidden Layer to Output Layer
    Dense_7 = keras.layers.Dense(units = 1, use_bias = False)
    Reshape_7 = keras.layers.Reshape(target_shape = (MAX_QUERY_WC,), name = 'Reshape_7')
    Softmax_7 = keras.layers.Softmax()
    query_weights = Dense_7(embedded_query)  # shape == [BATCH_SIZE, MAX_QUERY_WC, 1]
    query_weights = Reshape_7(query_weights)  # shape == [BATCH_SIZE, MAX_QUERY_WC]
    query_weights = Softmax_7(query_weights)  # shape == [BATCH_SIZE, MAX_QUERY_WC]
    
    # Output Layer
    Dot_8 = keras.layers.Dot(axes = [-1, -1], name = 'Dot_8')
    Sigmoid_8 = keras.layers.Activation(activation = 'sigmoid', name = 'Sigmoid_8')
    logits = Dot_8([query_weights, last_hidden_layer])  # shape == [BATCH_SIZE, 1]
    prediction = Sigmoid_8(logits)  # shape == [BATCH_SIZE, 1]
    
    model = keras.Model(inputs = [embedded_query, embedded_doc], outputs = prediction)
    
    RMSprop_Optimizer = keras.optimizers.RMSprop(lr = INITIAL_LR, decay = INITIAL_DECAY)
    model.compile(optimizer = RMSprop_Optimizer, loss = 'binary_crossentropy', metrics = ['acc'])
    model.save(filepath = LATEST_MODEL_PATH)
    
    return model
