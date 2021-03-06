from typing import List
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
from config import WORD2VEC_MODEL_PATH, VEC_SIZE


np.random.seed(seed = 0)


class EpochLogger(CallbackAny2Vec):
    """Callback to log information about training"""
    
    def __init__(self):
        self.epoch = 0
    
    def on_train_begin(self, model):
        print('Start training Word2Vec model')
    
    def on_epoch_begin(self, model):
        self.epoch += 1
        print(f'Epoch #{self.epoch} start')
    
    def on_epoch_end(self, model):
        print(f'Epoch #{self.epoch} end')
        print(f'Save model to {WORD2VEC_MODEL_PATH}')
        model.wv.save(WORD2VEC_MODEL_PATH)


def get_word2vec_model() -> KeyedVectors:
    try:
        return KeyedVectors.load(WORD2VEC_MODEL_PATH)
    except OSError:
        pass
    
    from config import DATA_FILE_PATH, WORKERS
    from dataset import query_doc_label_generator, cut_sentence
    
    sentences: List[List[str]] = []
    for dataset in DATA_FILE_PATH:
        for query, doc_list, _ in query_doc_label_generator(dataset = dataset):
            sentences.append(cut_sentence(sentence = query))
            sentences.extend(map(cut_sentence, doc_list))
    
    epoch_logger = EpochLogger()
    
    model: Word2Vec = Word2Vec(sentences = sentences, size = VEC_SIZE, min_count = 3, workers = WORKERS)
    model.train(sentences = sentences, total_examples = len(sentences), epochs = 50, callbacks = [epoch_logger])
    
    model.wv.save(WORD2VEC_MODEL_PATH)
    
    return model.wv


word2vec: KeyedVectors = get_word2vec_model()


def get_vectors(word_list: List[str]) -> List[np.ndarray]:
    vec_list: List[np.ndarray] = []
    for word in word_list:
        try:
            vec: np.ndarray = word2vec.get_vector(word = word)
        except KeyError:
            vec: np.ndarray = np.zeros(shape = VEC_SIZE, dtype = np.float32)
        vec_list.append(vec)
    return vec_list
