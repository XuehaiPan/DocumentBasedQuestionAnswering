import os
from typing import List, Set
import numpy as np
import keras
from config import DATA_DIR, BATCH_SIZE, WORKERS
from model import get_model_paths, build_network
from dataset import DataSequence


def save_predictions(predictions: np.ndarray, filename: str) -> None:
    with open(file = filename, mode = 'w', encoding = 'UTF-8') as file:
        for score in predictions.flatten():
            file.write(f'{score}\n')


def load_predictions(filename: str) -> np.ndarray:
    predictions: List[float] = []
    with open(file = filename, mode = 'r', encoding = 'UTF-8') as file:
        for line in map(str.strip, file):
            predictions.append(float(line))
    return np.array(predictions, dtype = np.float32)


def predict() -> None:
    validation_data = DataSequence(dataset = 'validation', batch_size = BATCH_SIZE,
                                   data_augmentation = False, return_target = False)
    test_data = DataSequence(dataset = 'test', batch_size = BATCH_SIZE,
                             data_augmentation = False, return_target = False)
    
    model_paths: List[str] = get_model_paths(sort_by = 'val_acc', reverse = True)
    top5_model_paths: Set[str] = set(model_paths[:5])
    model_paths: List[str] = get_model_paths(sort_by = 'epoch', reverse = False)
    
    top5_validation_predictions: List[np.ndarray] = []
    top5_test_predictions: List[np.ndarray] = []
    validation_predictions: np.ndarray = None
    test_predictions: np.ndarray = None
    for epoch, model_path in enumerate(model_paths, start = 1):
        print(f'load model {model_path}')
        model: keras.Model = build_network(model_path = model_path)
        validation_file: str = os.path.join(DATA_DIR, f'validation_score_epoch{epoch}.txt')
        test_file: str = os.path.join(DATA_DIR, f'test_score_epoch{epoch}.txt')
        if not os.path.exists(validation_file):
            print('predict on validation data')
            validation_predictions = model.predict_generator(generator = validation_data, verbose = 1,
                                                             workers = WORKERS, use_multiprocessing = True)
            validation_predictions = validation_predictions.flatten()
            save_predictions(predictions = validation_predictions, filename = validation_file)
        if model_path in top5_model_paths:
            if validation_predictions is None:
                validation_predictions = load_predictions(filename = validation_file)
            if not os.path.exists(test_file):
                print('predict on test data')
                test_predictions = model.predict_generator(generator = test_data, verbose = 1,
                                                           workers = WORKERS, use_multiprocessing = True)
                test_predictions = test_predictions.flatten()
                save_predictions(predictions = test_predictions, filename = test_file)
            else:
                test_predictions = load_predictions(filename = validation_file)
            top5_validation_predictions.append(validation_predictions)
            top5_test_predictions.append(test_predictions)
    validation_predictions = np.mean(top5_validation_predictions, axis = 0)
    test_predictions = np.mean(top5_test_predictions, axis = 0)
    validation_file: str = os.path.join(DATA_DIR, f'validation_score_ensemble.txt')
    test_file: str = os.path.join(DATA_DIR, f'test_score_epoch_ensemble.txt')
    save_predictions(predictions = validation_predictions, filename = validation_file)
    save_predictions(predictions = test_predictions, filename = test_file)


def main() -> None:
    predict()


if __name__ == '__main__':
    main()
