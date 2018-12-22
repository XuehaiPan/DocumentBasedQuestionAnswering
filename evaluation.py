import os
from typing import List
import pandas as pd
import keras
from dataset import DataSequence
from config import BATCH_SIZE, WORKERS, LOG_DIR
from model import get_model_paths, build_network


def evaluate_all_models() -> None:
    logs = pd.DataFrame(columns = ['epoch', 'acc', 'loss', 'val_loss', 'val_acc'])
    
    model_paths: List[str] = get_model_paths(sort_by = 'epoch')
    
    train_data = DataSequence(dataset = 'train', batch_size = BATCH_SIZE,
                              data_augmentation = True, return_target = True)
    validation_data = DataSequence(dataset = 'validation', batch_size = BATCH_SIZE,
                                   data_augmentation = False, return_target = True)
    for epoch, model_path in enumerate(model_paths, start = 1):
        print(f'epoch {epoch}')
        model: keras.Model = build_network(model_path = model_path)
        loss, acc = model.evaluate_generator(generator = train_data, verbose = 1,
                                             workers = WORKERS, use_multiprocessing = True)
        val_loss, val_acc = model.evaluate_generator(generator = validation_data, verbose = 1,
                                                     workers = WORKERS, use_multiprocessing = True)
        logs = logs.append({
            'epoch': epoch - 1,
            'acc': acc, 'loss': loss,
            'val_loss': val_loss, 'val_acc': val_acc
        }, ignore_index = True)
        
        logs.to_csv(os.path.join(LOG_DIR, 'evaluation_result.csv'), index = False)


def main() -> None:
    evaluate_all_models()


if __name__ == '__main__':
    main()
