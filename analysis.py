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


def predict() -> None:
    validation_data = DataSequence(dataset = 'validation', batch_size = BATCH_SIZE,
                                   data_augmentation = False, return_target = False)
    test_data = DataSequence(dataset = 'test', batch_size = BATCH_SIZE,
                             data_augmentation = False, return_target = False)
    
    model_paths: List[str] = get_model_paths(sort_by = 'val_acc', reverse = True)
    top5_model_paths: Set[str] = set(model_paths[:5])
    model_paths: List[str] = get_model_paths(sort_by = 'epoch', reverse = False)
    
    for epoch, model_path in enumerate(model_paths, start = 1):
        model: keras.Model = build_network(model_path = model_path)
        validation_file: str = os.path.join(DATA_DIR, f'validation_score_epoch{epoch}.txt')
        test_file: str = os.path.join(DATA_DIR, f'test_score_epoch{epoch}.txt')
        if not os.path.exists(validation_file):
            validation_predictions: np.ndarray = model.predict_generator(generator = validation_data, verbose = 1,
                                                                         workers = WORKERS, use_multiprocessing = True)
            save_predictions(predictions = validation_predictions, filename = validation_file)
        if not os.path.exists(test_file) and model_path in top5_model_paths:
            test_predictions: np.ndarray = model.predict_generator(generator = test_data, verbose = 1,
                                                                   workers = WORKERS, use_multiprocessing = True)
            save_predictions(predictions = test_predictions, filename = test_file)


def main() -> None:
    predict()


if __name__ == '__main__':
    main()
