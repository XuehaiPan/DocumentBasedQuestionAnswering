import os


data_dir = './data/'
data_file_path = {
    name: os.path.join(data_dir, '{}-set.data'.format(name))
    for name in ('train', 'validation', 'test')
}

# test data not given yet
del data_file_path['test']

POSITIVE = 1
NEGATIVE = 0


def split_line(line):
    return list(map(str.strip, line.split('\t')))


def quest_ans_label_generator(dataset):
    # add alias
    if 'valid' in dataset:
        dataset = 'validation'
    with open(file = data_file_path[dataset], encoding = 'UTF-8') as file:
        for i, line in enumerate(file, start = 1):
            split = split_line(line = line)
            if len(split) != 3:
                raise ValueError('Invalid data format.\n' +
                                 '  File \"{}\", line {}\n'.format(data_file_path[dataset], i) +
                                 '     original: \"{}\"\n     split: {}\n'.format(line.rstrip(), split))
            yield split


def main():
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
    
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(nrows = 3, ncols = 2, figsize = (12, 18))
    train_quest_seq_len, train_ans_seq_len = get_seq_len(dataset = 'train')
    valid_quest_seq_len, valid_ans_seq_len = get_seq_len(dataset = 'valid')
    
    color = plt.rcParamsDefault['axes.prop_cycle']
    color = iter(color)
    
    sns.distplot(train_quest_seq_len, color = next(color)['color'], kde = False, label = 'Question', ax = axes[0, 0])
    axes[0, 0].set_xlim(left = 0)
    axes[0, 0].set_xlabel('length')
    axes[0, 0].set_ylabel('frequency')
    axes[0, 0].set_title('Sequence Length in Training Data')
    axes[0, 0].legend()
    
    sns.distplot(train_ans_seq_len[train_ans_seq_len <= 400], color = next(color)['color'], kde = False, label = 'Answer', ax = axes[0, 1])
    axes[0, 1].set_xlim(left = 0, right = 400)
    axes[0, 1].set_xlabel('length')
    axes[0, 1].set_ylabel('frequency')
    axes[0, 1].set_title('Sequence Length in Training Data')
    axes[0, 1].legend()
    
    sns.distplot(valid_quest_seq_len, color = next(color)['color'], kde = False, label = 'Question', ax = axes[1, 0])
    axes[1, 0].set_xlim(left = 0)
    axes[1, 0].set_xlabel('length')
    axes[1, 0].set_ylabel('frequency')
    axes[1, 0].set_title('Sequence Length in Validation Data')
    axes[1, 0].legend()
    
    sns.distplot(valid_ans_seq_len[valid_ans_seq_len <= 400], color = next(color)['color'], kde = False, label = 'Answer', ax = axes[1, 1])
    axes[1, 1].set_xlim(left = 0, right = 400)
    axes[1, 1].set_xlabel('length')
    axes[1, 1].set_ylabel('frequency')
    axes[1, 1].set_title('Sequence Length in Validation Data')
    axes[1, 1].legend()
    
    quest_seq_len = np.concatenate([train_quest_seq_len, valid_quest_seq_len])
    ans_seq_len = np.concatenate([train_ans_seq_len, valid_ans_seq_len])
    
    sns.distplot(quest_seq_len, color = next(color)['color'], kde = False, label = 'Question', ax = axes[2, 0])
    axes[2, 0].set_xlim(left = 0)
    axes[2, 0].set_xlabel('length')
    axes[2, 0].set_ylabel('frequency')
    axes[2, 0].set_title('Sequence Length in All Data')
    axes[2, 0].legend()
    
    sns.distplot(ans_seq_len[ans_seq_len <= 400], color = next(color)['color'], kde = False, label = 'Answer', ax = axes[2, 1])
    axes[2, 1].set_xlim(left = 0, right = 400)
    axes[2, 1].set_xlabel('length')
    axes[2, 1].set_ylabel('frequency')
    axes[2, 1].set_title('Sequence Length in All Data')
    axes[2, 1].legend()
    
    fig.tight_layout()
    fig.savefig(fname = './data_dist.png')
    fig.show()


if __name__ == '__main__':
    main()
