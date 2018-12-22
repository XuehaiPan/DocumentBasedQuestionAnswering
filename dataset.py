import os
from typing import List, Tuple, Dict, Iterator, Union
import numpy as np
import keras
from jieba import Tokenizer
from config import POSITIVE, NEGATIVE, \
    VEC_SIZE, MAX_QUERY_WC, MAX_DOC_WC, BIN_NUM, \
    DATA_FILE_PATH, DICTIONARY_PATH, FIGURE_DIR


def split_line(line: str) -> List[str]:
    return list(map(str.strip, line.split('\t')))


def data_tuple_generator(dataset: str) -> Tuple[str, str, int]:
    assert dataset in ('train', 'valid', 'validation', 'test')
    if 'valid' in dataset:  # add alias
        dataset = 'validation'
    with open(file = DATA_FILE_PATH[dataset], mode = 'r', encoding = 'UTF-8') as file:
        file: Iterator[str] = filter(None, map(str.strip, file))
        for i, line in enumerate(file, start = 1):
            split: List[str] = split_line(line = line)
            try:
                query, doc, label = split
                yield query, doc, int(label)
            except ValueError:
                query, doc = split
                yield query, doc, None


def query_doc_label_generator(dataset: str) -> Tuple[str, List[str], List[int]]:
    cur_query: str = None
    doc_list: List[str] = []
    label_list: List[int] = []
    for query, doc, label in data_tuple_generator(dataset = dataset):
        if query == cur_query:
            doc_list.append(doc)
            label_list.append(label)
        else:
            if cur_query is not None:
                yield cur_query, doc_list, label_list
            cur_query = query
            doc_list = [doc]
            label_list = [label]
    if cur_query is not None:
        yield cur_query, doc_list, label_list


# tokenizer to cut sentence
tokenizer: Tokenizer = Tokenizer(dictionary = None)

if not os.path.exists(DICTIONARY_PATH):
    from shutil import copyfile
    
    
    # copy default dictionary
    tokenizer.initialize(dictionary = None)
    with tokenizer.get_dict_file() as default_dict:
        copyfile(src = default_dict.name, dst = DICTIONARY_PATH)
    
    del copyfile

# initialize tokenizer
tokenizer.initialize(dictionary = DICTIONARY_PATH)


def cut_sentence(sentence: str) -> List[str]:
    return list(tokenizer.cut(sentence = sentence, cut_all = False, HMM = True))


try:
    from word2vec import get_vectors
except ImportError:
    get_vectors = None


def get_normalized_vectors(word_list: List[str], epsilon: float = 1E-12) -> List[np.ndarray]:
    vec_list: List[np.ndarray] = get_vectors(word_list = word_list)
    for i, vec in enumerate(vec_list):
        norm: float = max(np.linalg.norm(vec), epsilon)
        vec_list[i] = vec / norm
    return vec_list


def sent2vec(word_list: List[str], max_wc: int) -> np.ndarray:
    embedded_sent: np.ndarray = np.zeros(shape = (max_wc, VEC_SIZE), dtype = np.float32)
    vec_list: List[np.ndarray] = get_normalized_vectors(word_list = word_list)
    for i, vec in zip(range(max_wc), vec_list):
        embedded_sent[i] = vec
    return embedded_sent


class DataSequence(keras.utils.Sequence):
    def __init__(self, dataset: str, batch_size: int,
                 data_augmentation: bool = False, return_target: bool = True) -> None:
        super().__init__()
        self.dataset: str = dataset
        self.batch_size: int = batch_size
        self.data_augmentation: bool = False
        self.return_target: bool = return_target
        
        self.queries: List[List[str]] = []
        self.doc_lists: List[List[List[str]]] = []
        self.label_lists: List[List[int]] = []
        for query, doc_list, label_list in query_doc_label_generator(dataset = dataset):
            self.queries.append(cut_sentence(sentence = query))
            self.doc_lists.append(list(map(cut_sentence, doc_list)))
            self.label_lists.append(label_list)
        
        if data_augmentation:
            self.do_data_augmentation()
        
        self.show_dataset_info()
    
    def __getitem__(self, index: int) -> Union[Tuple[List[np.ndarray], np.ndarray], List[np.ndarray]]:
        queries: List[List[str]] = self.queries[index * self.batch_size:(index + 1) * self.batch_size]
        doc_lists: List[List[List[str]]] = self.doc_lists[index * self.batch_size:(index + 1) * self.batch_size]
        label_lists: List[List[int]] = self.label_lists[index * self.batch_size:(index + 1) * self.batch_size]
        
        n_sample: int = sum(map(len, label_lists))
        embedded_queries: np.ndarray = np.zeros(shape = (n_sample, MAX_QUERY_WC, VEC_SIZE), dtype = np.float32)
        bin_sum: np.ndarray = np.zeros(shape = (n_sample, MAX_QUERY_WC, BIN_NUM), dtype = np.float32)
        
        s: int = 0
        for query, doc_list, label_list in zip(queries, doc_lists, label_lists):
            embedded_query: np.ndarray = sent2vec(word_list = query, max_wc = MAX_QUERY_WC)
            for doc, label in zip(doc_list, label_list):
                embedded_doc: np.ndarray = sent2vec(word_list = doc, max_wc = MAX_DOC_WC)
                qa_matrix: np.ndarray = np.tensordot(embedded_query, embedded_doc, axes = (-1, -1))
                for i in range(MAX_QUERY_WC):
                    for j in range(MAX_DOC_WC):
                        val: float = qa_matrix[i, j]
                        k: int = int((val + 1) * (BIN_NUM - 1) / 2)
                        bin_sum[s, i, k] += qa_matrix[i, j]
                s += 1
        
        if self.return_target:
            labels: np.ndarray = np.concatenate(label_lists).astype(dtype = np.float32)
            return [embedded_queries, bin_sum], labels
        else:
            return [embedded_queries, bin_sum]
    
    def __len__(self) -> int:
        return int(np.ceil(len(self.label_lists) / float(self.batch_size)))
    
    def do_data_augmentation(self) -> None:
        if not self.data_augmentation:
            for doc_list, label_list in zip(self.doc_lists, self.label_lists):
                n_doc: int = len(label_list)
                n_pos_doc: int = label_list.count(POSITIVE)
                n_neg_doc: int = label_list.count(NEGATIVE)
                try:
                    factor: int = n_neg_doc // n_pos_doc - 1
                except ZeroDivisionError:
                    pass
                else:
                    for i in range(n_doc):
                        if label_list[i] == POSITIVE:
                            doc_list.extend([doc_list[i]] * factor)
                            label_list.extend([POSITIVE] * factor)
        self.data_augmentation = True
    
    def show_dataset_info(self) -> None:
        n_query: int = len(self.queries)
        n_doc: int = sum(map(len, self.label_lists))
        n_pos_doc: int = sum(label_list.count(POSITIVE) for label_list in self.label_lists)
        n_neg_doc: int = sum(label_list.count(NEGATIVE) for label_list in self.label_lists)
        print(f'{{\n'
              f'    dataset: {self.dataset},\n'
              f'    query_num: {n_query},\n'
              f'    doc_num: {n_doc},\n'
              f'    positive_doc_num: {n_pos_doc},\n'
              f'    negative_doc_num: {n_neg_doc},\n'
              f'    average_doc_count_per_query: {n_doc / n_query:.2f},\n'
              f'    average_positive_doc_num_per_query: {n_pos_doc / n_query:.2f},\n'
              f'    average_negative_doc_num_per_query: {n_neg_doc / n_query:.2f},\n'
              f'    use_data_augmentation: {self.data_augmentation}\n'
              f'}}')


def draw_data_distribution() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    color = plt.rcParamsDefault['axes.prop_cycle']
    color = list(map(lambda c: c['color'], color)) * 5
    color = iter(color)
    
    def get_seq_len(dataset: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        query_wc_list: List[int] = []
        doc_wc_list: List[int] = []
        doc_cnt_list: List[int] = []
        longest_query: str = ''
        max_query_wc: int = 0
        longest_doc: str = ''
        longest_doc_query: str = ''
        max_doc_wc: int = 0
        for query, doc_list, _ in query_doc_label_generator(dataset = dataset):
            query_wc: int = len(cut_sentence(sentence = query))
            query_wc_list.append(query_wc)
            doc_cnt_list.append(len(doc_list))
            if query_wc > max_query_wc:
                longest_query, max_query_wc = query, query_wc
            for doc in doc_list:
                doc_wc: int = len(cut_sentence(sentence = doc))
                doc_wc_list.append(doc_wc)
                if doc_wc > max_doc_wc:
                    longest_doc, longest_doc_query, max_doc_wc = doc, query, doc_wc
        print(f'{{\n'
              f'    dataset: {dataset},\n'
              f'    \n'
              f'    longest_query: \'{longest_query}\',\n'
              f'    max_query_word_count: {max_query_wc},\n'
              f'    \n'
              f'    longest_doc: \'{longest_doc}\',\n'
              f'    query_of_longest_doc: \'{longest_doc_query}\',\n'
              f'    max_doc_word_count: {max_doc_wc}\n'
              f'}}\n')
        return np.array(query_wc_list, dtype = np.int32), np.array(doc_wc_list, dtype = np.int32), \
               np.array(doc_cnt_list, dtype = np.int32)
    
    def plot_dist(dataset: str,
                  query_wc: np.ndarray, doc_wc: np.ndarray, doc_cnt: np.ndarray,
                  query_ax: plt.Axes, doc_ax: plt.Axes, doc_cnt_ax: plt.Axes) -> None:
        sns.distplot(query_wc[query_wc <= MAX_QUERY_WC], bins = min(query_wc.max(), MAX_QUERY_WC),
                     kde = True, kde_kws = {'label': 'kernel density estimation'},
                     color = next(color), label = 'data', ax = query_ax)
        sns.distplot(doc_wc[doc_wc <= MAX_DOC_WC], bins = MAX_DOC_WC,
                     kde = True, kde_kws = {'label': 'kernel density estimation'},
                     color = next(color), label = 'data', ax = doc_ax)
        sns.distplot(doc_cnt, bins = doc_cnt.max(),
                     kde = True, kde_kws = {'alpha': 0},
                     color = next(color), label = 'data', ax = doc_cnt_ax)
        for q in (0.25, 0.50, 0.75):
            query_ax.axvline(x = np.quantile(query_wc, q = q),
                             linestyle = '-.', color = 'black', alpha = 0.5)
            doc_ax.axvline(x = np.quantile(doc_wc, q = q),
                           linestyle = '-.', color = 'black', alpha = 0.5)
        query_ax.set_xlim(left = 0)
        doc_ax.set_xlim(left = 0, right = MAX_DOC_WC)
        doc_cnt_ax.set_xlim(left = 0)
        query_ax.set_title(label = f'Query Length ({dataset})')
        doc_ax.set_title(label = f'Doc Length ({dataset})')
        doc_cnt_ax.set_title(label = f'Doc Count Per Query ({dataset})')
        for ax in (query_ax, doc_ax, doc_cnt_ax):
            ax.set_xlabel(xlabel = 'Word Count')
            ax.set_ylabel(ylabel = 'Relative Frequency')
            ax.legend()
        doc_cnt_ax.set_xlabel(xlabel = 'Doc Count')
    
    fig: plt.Figure
    axes: Dict[Tuple[int, int], plt.Axes]
    fig, axes = plt.subplots(nrows = 4, ncols = 3, figsize = (18, 18), dpi = 250)
    train_query_wc, train_doc_wc, train_doc_cnt = get_seq_len(dataset = 'train')
    valid_query_wc, valid_doc_wc, valid_doc_cnt = get_seq_len(dataset = 'validation')
    test_query_wc, test_doc_wc, test_doc_cnt = get_seq_len(dataset = 'test')
    
    plot_dist(dataset = 'train',
              query_wc = train_query_wc, doc_wc = train_doc_wc, doc_cnt = train_doc_cnt,
              query_ax = axes[0, 0], doc_ax = axes[0, 1], doc_cnt_ax = axes[0, 2])
    
    plot_dist(dataset = 'validation',
              query_wc = valid_query_wc, doc_wc = valid_doc_wc, doc_cnt = valid_doc_cnt,
              query_ax = axes[1, 0], doc_ax = axes[1, 1], doc_cnt_ax = axes[1, 2])
    
    plot_dist(dataset = 'test',
              query_wc = test_query_wc, doc_wc = test_doc_wc, doc_cnt = test_doc_cnt,
              query_ax = axes[2, 0], doc_ax = axes[2, 1], doc_cnt_ax = axes[2, 2])
    
    all_query_wc: np.ndarray = np.concatenate([train_query_wc, valid_query_wc, test_query_wc])
    all_doc_wc: np.ndarray = np.concatenate([train_doc_wc, valid_doc_wc, test_doc_wc])
    all_doc_cnt: np.ndarray = np.concatenate([train_doc_cnt, valid_doc_cnt, test_doc_cnt])
    
    plot_dist(dataset = 'all',
              query_wc = all_query_wc, doc_wc = all_doc_wc, doc_cnt = all_doc_cnt,
              query_ax = axes[3, 0], doc_ax = axes[3, 1], doc_cnt_ax = axes[3, 2])
    
    fig.tight_layout()
    fig.savefig(fname = os.path.join(FIGURE_DIR, 'data_dist.png'))
    fig.show()


def main() -> None:
    draw_data_distribution()


if __name__ == '__main__':
    main()
