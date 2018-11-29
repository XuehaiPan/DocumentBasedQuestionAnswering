from typing import List
from tensorflow import keras
from DataSet import quest_ans_label_generator
from Config import BATCH_SIZE, \
    MODEL_FMT_STR, MODEL_FILE_PATTERN, LATEST_MODEL_PATH, \
    LOG_DIR, LOG_FILE_PATH
from Model import get_model_paths, build_network


def train(epochs: int) -> None:
    # TODO: get data
    x_train, y_train = None, None
    x_valid, y_valid = None, None
    
    tensorBoard = keras.callbacks.TensorBoard(log_dir = LOG_DIR,
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
    earlyStopping = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                                  patience = 5,
                                                  verbose = 1)
    
    try:
        model_paths: List[str] = get_model_paths(sortby = 'epoch', reverse = False)
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
                  callbacks = [tensorBoard, csvLogger, checkpoint, checkpointLatest, earlyStopping],
                  workers = 4, use_multiprocessing = True)
    except KeyboardInterrupt:
        pass
    finally:
        model.save(filepath = LATEST_MODEL_PATH)


def main() -> None:
    train(epochs = 40)


if __name__ == '__main__':
    main()
