from typing import List
from tensorflow import keras
from dataset import query_doc_label_generator
from .config import BATCH_SIZE, \
    MODEL_FMT_STR, MODEL_FILE_PATTERN, LATEST_MODEL_PATH, \
    LOG_DIR, LOG_FILE_PATH
from .model import get_model_paths, build_network


class MyTensorBoard(keras.callbacks.TensorBoard):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        # log learning rate
        logs.update(lr = keras.backend.eval(self.model.optimizer.lr))
        super().on_epoch_end(epoch, logs)


def train(epochs: int) -> None:
    # TODO: get data
    raise NotImplementedError
    x_train, y_train = None, None
    x_valid, y_valid = None, None
    
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
                                                 verbose = 1)
    checkpointLatest = keras.callbacks.ModelCheckpoint(filepath = LATEST_MODEL_PATH,
                                                       monitor = 'val_acc',
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
        model.save(filepath = LATEST_MODEL_PATH)


def main() -> None:
    train(epochs = 50)


if __name__ == '__main__':
    main()
