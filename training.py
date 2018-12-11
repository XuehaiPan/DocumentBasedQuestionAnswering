from typing import List, Tuple
import numpy as np
from tensorflow import keras
from dataset import data_tuple_generator, cut_sentence
from word2vec import get_vectors
from config import VEC_SIZE, MAX_QUERY_WC, MAX_DOC_WC, BATCH_SIZE, \
    MODEL_FMT_STR, MODEL_FILE_PATTERN, LATEST_MODEL_PATH, \
    LOG_DIR, LOG_FILE_PATH
from model import get_model_paths, build_network


class MyTensorBoard(keras.callbacks.TensorBoard):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        # log learning rate
        try:
            logs.update(lr = keras.backend.eval(self.model.optimizer.lr))
        except AttributeError:
            pass
        super().on_epoch_end(epoch, logs)


def sent2vec(sentence: str, max_wc: int) -> np.ndarray:
    embedded: np.ndarray = np.zeros(shape = (max_wc, VEC_SIZE), dtype = np.float32)
    word_list: List[str] = cut_sentence(sentence = sentence)
    vec_list: List[np.ndarray] = get_vectors(word_list = word_list)
    for i, vec in zip(range(max_wc), vec_list):
        embedded[i] = vec
    return embedded


def get_data(dataset: str) -> Tuple[List[np.ndarray], np.ndarray]:
    query_vectors: List[np.ndarray] = []
    doc_vectors: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for query, doc, label in data_tuple_generator(dataset = dataset):
        query_vectors.append(sent2vec(sentence = query, max_wc = MAX_QUERY_WC))
        doc_vectors.append(sent2vec(sentence = doc, max_wc = MAX_DOC_WC))
        labels.append(label)
    
    query_vectors: np.ndarray = np.array(query_vectors, dtype = np.float32)
    doc_vectors: np.ndarray = np.array(doc_vectors, dtype = np.float32)
    labels: np.ndarray = np.array(labels, dtype = np.float32)
    return [query_vectors, doc_vectors], labels


def train(epochs: int) -> None:
    x_train, y_train = get_data(dataset = 'train')
    x_valid, y_valid = get_data(dataset = 'validation')
    
    tensorBoard = MyTensorBoard(log_dir = LOG_DIR,
                                histogram_freq = 0,
                                batch_size = BATCH_SIZE,
                                write_graph = True,
                                write_grads = True,
                                write_images = True)
    csvLogger = keras.callbacks.CSVLogger(filename = LOG_FILE_PATH,
                                          append = True)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath = MODEL_FMT_STR,
                                                 monitor = 'val_acc',
                                                 save_weights_only = True,
                                                 verbose = 1)
    checkpointLatest = keras.callbacks.ModelCheckpoint(filepath = LATEST_MODEL_PATH,
                                                       monitor = 'val_acc',
                                                       save_weights_only = True,
                                                       verbose = 1)
    terminateOnNaN = keras.callbacks.TerminateOnNaN()
    earlyStopping = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                                  patience = 5,
                                                  verbose = 1)
    
    try:
        model_paths: List[str] = get_model_paths(sort_by = 'epoch', reverse = False)
        initial_model_path: str = model_paths[-1]
        initial_epoch: int = int(MODEL_FILE_PATTERN.match(string = initial_model_path).group('epoch'))
    except IndexError:
        initial_epoch: int = 0
        initial_model_path: str = LATEST_MODEL_PATH
    
    model: keras.Model = build_network(model_path = initial_model_path)
    model.summary()
    
    try:
        print(f'initial_epoch = {initial_epoch}')
        print(f'initial_model_path = {initial_model_path}')
        model.fit(x = x_train, y = y_train,
                  batch_size = BATCH_SIZE, epochs = epochs, initial_epoch = initial_epoch,
                  validation_data = (x_valid, y_valid), shuffle = True,
                  callbacks = [tensorBoard, csvLogger, checkpoint, checkpointLatest, terminateOnNaN, earlyStopping],
                  workers = 4, use_multiprocessing = True)
    except KeyboardInterrupt:
        pass
    finally:
        print(f'save current model to {LATEST_MODEL_PATH}')
        model.save_weights(filepath = LATEST_MODEL_PATH)


def main() -> None:
    train(epochs = 80)


if __name__ == '__main__':
    main()
