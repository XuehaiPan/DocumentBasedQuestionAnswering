from typing import List
import keras
from dataset import DataSequence
from config import BATCH_SIZE, WORKERS, \
    MODEL_FMT_STR, MODEL_FILE_PATTERN, LATEST_MODEL_PATH, \
    LOG_DIR, LOG_FILE_PATH
from model import get_model_paths, build_network


class MyTensorBoard(keras.callbacks.TensorBoard):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        # log learning rate
        logs = logs or {}
        try:
            logs.update(lr = keras.backend.eval(self.model.optimizer.lr))
        except AttributeError:
            pass
        super().on_epoch_end(epoch, logs)


class MyModelCheckpoint(keras.callbacks.ModelCheckpoint):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        try:
            super().on_epoch_end(epoch = epoch, logs = logs)
        except OSError:
            pass


def train(epochs: int) -> None:
    train_data = DataSequence(dataset = 'train', batch_size = BATCH_SIZE, data_augmentation = True)
    validation_data = DataSequence(dataset = 'validation', batch_size = BATCH_SIZE, data_augmentation = False)
    
    tensorBoard = MyTensorBoard(log_dir = LOG_DIR,
                                histogram_freq = 0,
                                batch_size = BATCH_SIZE,
                                write_graph = True,
                                write_grads = True,
                                write_images = True,
                                update_freq = 'batch')
    csvLogger = keras.callbacks.CSVLogger(filename = LOG_FILE_PATH,
                                          append = True)
    modelCheckpointEpoch = MyModelCheckpoint(filepath = MODEL_FMT_STR,
                                             monitor = 'val_acc',
                                             verbose = 1)
    modelCheckpointLatest = MyModelCheckpoint(filepath = LATEST_MODEL_PATH,
                                              monitor = 'val_acc',
                                              verbose = 1)
    terminateOnNaN = keras.callbacks.TerminateOnNaN()
    earlyStopping = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                                  patience = 5,
                                                  verbose = 1)
    
    callbacks: List[keras.callbacks.Callback] = [
        tensorBoard, csvLogger,
        modelCheckpointEpoch, modelCheckpointLatest,
        terminateOnNaN, earlyStopping
    ]
    
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
        print(f'workers = {WORKERS}')
        model.fit_generator(generator = train_data,
                            epochs = epochs, initial_epoch = initial_epoch,
                            validation_data = validation_data, shuffle = True,
                            callbacks = callbacks,
                            workers = WORKERS, use_multiprocessing = True)
    except KeyboardInterrupt:
        pass
    finally:
        print(f'save current model to {LATEST_MODEL_PATH}')
        model.save(filepath = LATEST_MODEL_PATH)


def main() -> None:
    train(epochs = 80)


if __name__ == '__main__':
    main()
