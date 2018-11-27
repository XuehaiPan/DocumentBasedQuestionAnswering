import os


data_dir = './data/'
data_file_path = {
    name: os.path.join(data_dir, '{}-set.data'.format(name))
    for name in ('train', 'validation', 'test')
}

# test data not given yet
del data_file_path['test']

# labels
POSITIVE = 1
NEGATIVE = 0
