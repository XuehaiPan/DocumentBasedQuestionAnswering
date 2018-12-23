import os
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import keras
from config import LOG_FILE_PATH, FIGURE_DIR
from model import get_model_paths, build_network


def draw_training_logs() -> None:
    logs: pd.DataFrame = pd.read_csv(LOG_FILE_PATH)
    logs['epoch'] += 1
    
    print(logs)
    
    fig: plt.Figure
    axes: List[plt.Axes]
    fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (12, 12))
    
    axes[0].plot(logs['epoch'], logs['acc'], label = 'train')
    axes[0].plot(logs['epoch'], logs['val_acc'], label = 'validation')
    axes[0].legend()
    axes[0].set_xlim(left = 0)
    axes[0].grid(axis = 'y')
    axes[0].set_xlabel(xlabel = 'epoch')
    axes[0].set_ylabel(ylabel = 'accuracy')
    axes[0].set_title(label = 'Accuracy')
    
    axes[1].plot(logs['epoch'], logs['loss'], label = 'train')
    axes[1].plot(logs['epoch'], logs['val_loss'], label = 'validation')
    axes[1].legend()
    axes[1].set_xlim(left = 0)
    axes[1].grid(axis = 'y')
    axes[1].set_xlabel(xlabel = 'epoch')
    axes[1].set_ylabel(ylabel = 'loss')
    axes[1].set_title(label = 'Loss')
    
    fig.tight_layout()
    fig.savefig(fname = os.path.join(FIGURE_DIR, 'training_logs.png'))
    fig.show()


def show_weight() -> None:
    def update(epoch: int) -> None:
        w: np.ndarray = weights[epoch]
        im.set_array(w.transpose())
        ax.set_xlabel('bin')
        ax.set_ylabel('weight index')
        ax.set_xticks(ticks = np.linspace(start = 0, stop = 200, num = 9, endpoint = True))
        ax.set_xticklabels(labels = np.linspace(start = -1, stop = 1, num = 9, endpoint = True))
        ax.set_yticks(ticks = [0, 127])
        ax.set_title(f'epoch {epoch + 1}')
        fig.tight_layout()
    
    model_paths: List[str] = get_model_paths(sort_by = 'epoch', reverse = False)
    
    weights: List[np.ndarray] = []
    for epoch, model_path in enumerate(model_paths, start = 1):
        model: keras.Model = build_network(model_path = model_path)
        Dense_1: keras.layers.Layer = model.get_layer(name = 'Dense_1')
        w1, b1 = Dense_1.get_weights()
        weights.append(w1)
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, squeeze = True, figsize = (12, 6), dpi = 150)
    im = ax.imshow(np.zeros(shape = weights[0].transpose().shape),
                   cmap = 'seismic', animated = True, vmin = -1.1, vmax = 1.1)
    cb = fig.colorbar(im, ax = ax)
    cb.set_label('weight value')
    fig.tight_layout()
    
    anim = FuncAnimation(fig = fig, func = update, frames = len(weights), interval = 1000)
    
    anim.save(filename = os.path.join(FIGURE_DIR, 'Dense_1_weight.mp4'))
    fig.show()


def main() -> None:
    draw_training_logs()
    show_weight()


if __name__ == '__main__':
    main()
