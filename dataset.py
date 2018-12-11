import os
from typing import List, Tuple, Dict, Iterator
from jieba import Tokenizer
from config import DATA_FILE_PATH, DICTIONARY_PATH, FIGURE_DIR


def split_line(line: str) -> List[str]:
    return list(map(str.strip, line.split('\t')))


def data_tuple_generator(dataset: str) -> Tuple[str, str, int]:
    with open(file = DATA_FILE_PATH[dataset], mode = 'r', encoding = 'UTF-8') as file:
        file: Iterator[str] = filter(None, map(str.strip, file))
        for i, line in enumerate(file, start = 1):
            split: List[str] = split_line(line = line)
            try:
                query, doc, label = split
                yield query, doc, int(label)
            except ValueError:  # assert len(split) == 3
                raise ValueError('Invalid data format.\n'
                                 f'  File \"{DATA_FILE_PATH[dataset]}\", line {i}\n'
                                 f'     original: \"{line}\"\n'
                                 f'     split: {split}\n')


def query_doc_label_generator(dataset: str) -> Tuple[str, List[str], List[int]]:
    assert dataset in ('train', 'valid', 'validation', 'test')
    if 'valid' in dataset:  # add alias
        dataset = 'validation'
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
            doc_list = []
            label_list = []
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


def draw_data_distribution() -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    color = plt.rcParamsDefault['axes.prop_cycle']
    color = iter(map(lambda c: c['color'], color))
    
    def get_seq_len(dataset: str) -> Tuple[np.ndarray, np.ndarray]:
        query_seq_len: List[int] = []
        doc_seq_len: List[int] = []
        longest_query = ''
        max_query_len = 0
        longest_doc = ''
        longest_doc_query = ''
        max_doc_len = 0
        for query, doc_list, _ in query_doc_label_generator(dataset = dataset):
            query_len = len(cut_sentence(sentence = query))
            query_seq_len.append(query_len)
            if query_len > max_query_len:
                longest_query, max_query_len = query, query_len
            for doc in doc_list:
                doc_len = len(cut_sentence(sentence = doc))
                doc_seq_len.append(doc_len)
                if doc_len > max_doc_len:
                    longest_doc, longest_doc_query, max_doc_len = doc, query, doc_len
        print(f'{{\n'
              f'    dataset: {dataset}\n'
              f'    \n'
              f'    longest_query: \'{longest_query}\',\n'
              f'    cut_longest_query: {cut_sentence(sentence = longest_query)}\n'
              f'    max_query_word_count: {max_query_len}\n'
              f'    \n'
              f'    longest_doc: \'{longest_doc}\',\n'
              f'    cut_longest_doc: {cut_sentence(sentence = longest_doc)}\n'
              f'    query_of_longest_doc: \'{longest_doc_query}\',\n'
              f'    max_doc_word_count: {max_doc_len}\n'
              f'}}\n')
        return np.array(query_seq_len, dtype = np.int32), np.array(doc_seq_len, dtype = np.int32)
    
    def plot_dist(dataset: str,
                  query_seq_len: np.ndarray, doc_seq_len: np.ndarray,
                  query_ax: plt.Axes, doc_ax: plt.Axes) -> None:
        max_doc_len: int = 160
        assert np.quantile(doc_seq_len, q = 0.99) <= max_doc_len
        sns.distplot(query_seq_len, bins = query_seq_len.max(), color = next(color),
                     kde = True, kde_kws = {'label': 'kernel density estimation'},
                     label = 'data', ax = query_ax)
        sns.distplot(doc_seq_len[doc_seq_len <= max_doc_len], color = next(color),
                     kde = True, kde_kws = {'label': 'kernel density estimation'},
                     label = 'data', ax = doc_ax)
        for q in (0.25, 0.50, 0.75):
            query_ax.axvline(x = np.quantile(query_seq_len, q = q),
                             linestyle = '-.', color = 'black', alpha = 0.5)
            doc_ax.axvline(x = np.quantile(doc_seq_len, q = q),
                           linestyle = '-.', color = 'black', alpha = 0.5)
        query_ax.set_xlim(left = 0)
        doc_ax.set_xlim(left = 0, right = max_doc_len)
        query_ax.set_title(label = f'Query Length ({dataset})')
        doc_ax.set_title(label = f'Doc Length ({dataset})')
        for ax in (query_ax, doc_ax):
            ax.set_xlabel(xlabel = 'Word Count')
            ax.set_ylabel(ylabel = 'Relative Frequency')
            ax.legend()
    
    fig: plt.Figure
    axes: Dict[Tuple[int, int], plt.Axes]
    fig, axes = plt.subplots(nrows = 3, ncols = 2, figsize = (12, 18), dpi = 250)
    train_query_seq_len, train_doc_seq_len = get_seq_len(dataset = 'train')
    valid_query_seq_len, valid_doc_seq_len = get_seq_len(dataset = 'validation')
    
    plot_dist(dataset = 'train',
              query_seq_len = train_query_seq_len, doc_seq_len = train_doc_seq_len,
              query_ax = axes[0, 0], doc_ax = axes[0, 1])
    
    plot_dist(dataset = 'validation',
              query_seq_len = valid_query_seq_len, doc_seq_len = valid_doc_seq_len,
              query_ax = axes[1, 0], doc_ax = axes[1, 1])
    
    all_query_seq_len: np.ndarray = np.concatenate([train_query_seq_len, valid_query_seq_len])
    all_doc_seq_len: np.ndarray = np.concatenate([train_doc_seq_len, valid_doc_seq_len])
    
    plot_dist(dataset = 'all',
              query_seq_len = all_query_seq_len, doc_seq_len = all_doc_seq_len,
              query_ax = axes[2, 0], doc_ax = axes[2, 1])
    
    fig.tight_layout()
    fig.savefig(fname = os.path.join(FIGURE_DIR, 'data_dist.png'))
    fig.show()


def main() -> None:
    draw_data_distribution()


if __name__ == '__main__':
    main()
