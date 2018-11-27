from Config import data_file_path


def split_line(line):
    return list(map(str.strip, line.split('\t')))


def quest_ans_label_generator(dataset):
    if 'valid' in dataset:
        dataset = 'validation'  # add alias
    with open(file = data_file_path[dataset], encoding = 'UTF-8') as file:
        for i, line in enumerate(file, start = 1):
            split = split_line(line = line)
            if len(split) != 3:
                raise ValueError('Invalid data format.\n'
                                 f'  File \"{data_file_path[dataset]}\", line {i}\n'
                                 f'     original: \"{line.rstrip()}\"\n'
                                 f'     split: {split}\n')
            yield split


# add alias
data_generator = quest_ans_label_generator


def draw_data_distribution():
    def get_seq_len(dataset):
        quest_seq_len = []
        ans_seq_len = []
        questions = set()
        for quest, ans, label in quest_ans_label_generator(dataset = dataset):
            if quest not in questions:
                quest_seq_len.append(len(quest))
                questions.add(quest)
            ans_seq_len.append(len(ans))
        return np.array(quest_seq_len, dtype = np.int32), np.array(ans_seq_len, dtype = np.int32)
    
    def plot_dist(dataset, quest_seq_len, ans_seq_len, quest_ax, ans_ax):
        sns.distplot(quest_seq_len, color = next(color)['color'],
                     kde = True, kde_kws = {'label': 'kernel density estimation'},
                     label = 'data', ax = quest_ax)
        sns.distplot(ans_seq_len[ans_seq_len <= 400], color = next(color)['color'],
                     kde = True, kde_kws = {'label': 'kernel density estimation'},
                     label = 'data', ax = ans_ax)
        for q in (0.25, 0.50, 0.75):
            quest_ax.axvline(x = np.quantile(quest_seq_len, q), linestyle = '-.', color = 'black', alpha = 0.5)
            ans_ax.axvline(x = np.quantile(ans_seq_len, q), linestyle = '-.', color = 'black', alpha = 0.5)
        quest_ax.set_xlim(left = 0)
        ans_ax.set_xlim(left = 0, right = 400)
        quest_ax.set_title(label = f'Question Length ({dataset})')
        ans_ax.set_title(label = f'Answer Length ({dataset})')
        for ax in (quest_ax, ans_ax):
            ax.set_xlabel(xlabel = 'length')
            ax.set_ylabel(ylabel = 'probability')
            ax.legend()
    
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(nrows = 3, ncols = 2, figsize = (12, 18))
    train_quest_seq_len, train_ans_seq_len = get_seq_len(dataset = 'train')
    valid_quest_seq_len, valid_ans_seq_len = get_seq_len(dataset = 'valid')
    
    color = plt.rcParamsDefault['axes.prop_cycle']
    color = iter(color)
    
    plot_dist(dataset = 'train',
              quest_seq_len = train_quest_seq_len, ans_seq_len = train_ans_seq_len,
              quest_ax = axes[0, 0], ans_ax = axes[0, 1])
    
    plot_dist(dataset = 'validation',
              quest_seq_len = valid_quest_seq_len, ans_seq_len = valid_ans_seq_len,
              quest_ax = axes[1, 0], ans_ax = axes[1, 1])
    
    quest_seq_len = np.concatenate([train_quest_seq_len, valid_quest_seq_len])
    ans_seq_len = np.concatenate([train_ans_seq_len, valid_ans_seq_len])
    
    plot_dist(dataset = 'all',
              quest_seq_len = quest_seq_len, ans_seq_len = ans_seq_len,
              quest_ax = axes[2, 0], ans_ax = axes[2, 1])
    
    fig.tight_layout()
    fig.savefig(fname = './data_dist.png')
    fig.show()


def main():
    draw_data_distribution()


if __name__ == '__main__':
    main()
