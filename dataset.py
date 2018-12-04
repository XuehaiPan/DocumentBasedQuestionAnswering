import os
from typing import List, Tuple, Set, Dict, Iterator
from jieba import Tokenizer
from config import DATA_FILE_PATH, DICTIONARY_PATH, FIGURE_DIR


def quest_ans_label_generator(dataset: str) -> Tuple[str, List[str], List[int]]:
    def split_line(line: str) -> List[str]:
        return list(map(str.strip, line.split('\t')))
    
    def data_tuple_generator(dataset: str) -> Tuple[str, str, int]:
        with open(file = DATA_FILE_PATH[dataset], mode = 'r', encoding = 'UTF-8') as file:
            file: Iterator[str] = filter(None, map(str.strip, file))
            for i, line in enumerate(file, start = 1):
                split: List[str] = split_line(line = line)
                try:
                    quest, ans, label = split
                    yield quest, ans, int(label)
                except ValueError:  # assert len(split) == 3
                    raise ValueError('Invalid data format.\n'
                                     f'  File \"{DATA_FILE_PATH[dataset]}\", line {i}\n'
                                     f'     original: \"{line}\"\n'
                                     f'     split: {split}\n')
    
    assert dataset in ('train', 'valid', 'validation', 'test')
    if 'valid' in dataset:  # add alias
        dataset = 'validation'
    cur_quest: str = None
    ans_list: List[str] = []
    label_list: List[int] = []
    for quest, ans, label in data_tuple_generator(dataset = dataset):
        if quest == cur_quest:
            ans_list.append(ans)
            label_list.append(label)
        else:
            if cur_quest is not None:
                yield cur_quest, ans_list, label_list
            cur_quest = quest
            ans_list = []
            label_list = []
    if cur_quest is not None:
        yield cur_quest, ans_list, label_list


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
    
    def get_seq_len(dataset: str) -> Tuple[np.ndarray, np.ndarray]:
        quest_seq_len: List[int] = []
        ans_seq_len: List[int] = []
        quest_set: Set[str] = set()
        longest_quest = ''
        max_quest_len = 0
        longest_ans = ''
        longest_ans_quest = ''
        max_ans_len = 0
        for quest, ans_list, _ in quest_ans_label_generator(dataset = dataset):
            quest_len = len(cut_sentence(sentence = quest))
            quest_seq_len.append(quest_len)
            if quest_len > max_quest_len:
                longest_quest, max_quest_len = quest, quest_len
                quest_set.add(quest)
            for ans in ans_list:
                ans_len = len(cut_sentence(sentence = ans))
                ans_seq_len.append(ans_len)
                if ans_len > max_ans_len:
                    longest_ans, longest_ans_quest, max_ans_len = ans, quest, ans_len
        print(f'{{\n'
              f'    dataset: {dataset}\n'
              f'    \n'
              f'    longest_question: \'{longest_quest}\',\n'
              f'    cut_longest_question: {cut_sentence(sentence = longest_quest)}\n'
              f'    max_question_word_count: {max_quest_len}\n'
              f'    \n'
              f'    longest_answer: \'{longest_ans}\',\n'
              f'    cut_longest_answer: {cut_sentence(sentence = longest_ans)}\n'
              f'    question_of_longest_answer: \'{longest_ans_quest}\',\n'
              f'    max_answer_word_count: {max_ans_len}\n'
              f'}}\n')
        return np.array(quest_seq_len, dtype = np.int32), np.array(ans_seq_len, dtype = np.int32)
    
    def plot_dist(dataset: str,
                  quest_seq_len: np.ndarray, ans_seq_len: np.ndarray,
                  quest_ax: plt.Axes, ans_ax: plt.Axes) -> None:
        max_ans_len: int = 160
        assert np.quantile(ans_seq_len, q = 0.99) <= max_ans_len
        sns.distplot(quest_seq_len, bins = quest_seq_len.max(), color = next(color),
                     kde = True, kde_kws = {'label': 'kernel density estimation'},
                     label = 'data', ax = quest_ax)
        sns.distplot(ans_seq_len[ans_seq_len <= max_ans_len], color = next(color),
                     kde = True, kde_kws = {'label': 'kernel density estimation'},
                     label = 'data', ax = ans_ax)
        for q in (0.25, 0.50, 0.75):
            quest_ax.axvline(x = np.quantile(quest_seq_len, q = q),
                             linestyle = '-.', color = 'black', alpha = 0.5)
            ans_ax.axvline(x = np.quantile(ans_seq_len, q = q),
                           linestyle = '-.', color = 'black', alpha = 0.5)
        quest_ax.set_xlim(left = 0)
        ans_ax.set_xlim(left = 0, right = max_ans_len)
        quest_ax.set_title(label = f'Question Length ({dataset})')
        ans_ax.set_title(label = f'Answer Length ({dataset})')
        for ax in (quest_ax, ans_ax):
            ax.set_xlabel(xlabel = 'Word Count')
            ax.set_ylabel(ylabel = 'Relative Frequency')
            ax.legend()
    
    fig: plt.Figure
    axes: Dict[Tuple[int, int], plt.Axes]
    fig, axes = plt.subplots(nrows = 3, ncols = 2, figsize = (12, 18), dpi = 250)
    train_quest_seq_len, train_ans_seq_len = get_seq_len(dataset = 'train')
    valid_quest_seq_len, valid_ans_seq_len = get_seq_len(dataset = 'validation')
    
    color = plt.rcParamsDefault['axes.prop_cycle']
    color = iter(map(lambda c: c['color'], color))
    
    plot_dist(dataset = 'train',
              quest_seq_len = train_quest_seq_len, ans_seq_len = train_ans_seq_len,
              quest_ax = axes[0, 0], ans_ax = axes[0, 1])
    
    plot_dist(dataset = 'validation',
              quest_seq_len = valid_quest_seq_len, ans_seq_len = valid_ans_seq_len,
              quest_ax = axes[1, 0], ans_ax = axes[1, 1])
    
    all_quest_seq_len: np.ndarray = np.concatenate([train_quest_seq_len, valid_quest_seq_len])
    all_ans_seq_len: np.ndarray = np.concatenate([train_ans_seq_len, valid_ans_seq_len])
    
    plot_dist(dataset = 'all',
              quest_seq_len = all_quest_seq_len, ans_seq_len = all_ans_seq_len,
              quest_ax = axes[2, 0], ans_ax = axes[2, 1])
    
    fig.tight_layout()
    fig.savefig(fname = os.path.join(FIGURE_DIR, 'data_dist.png'))
    fig.show()


def main() -> None:
    draw_data_distribution()


if __name__ == '__main__':
    main()
