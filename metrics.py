from typing import List, Set, Dict
from config import DATA_FILE_PATH
from dataset import query_doc_label_generator


QUERY_NUM: Dict[str, int] = {}
for dataset in DATA_FILE_PATH:
    query_set: Set[str] = set(map(lambda data_tuple: data_tuple[0],
                                  query_doc_label_generator(dataset = dataset)))
    QUERY_NUM[dataset] = len(query_set)

####################################################
# Unknown,,,
QUERY_CORRECT_NUM: Dict[str, int]
FINE_LABEL_DICT: Dict[str, List[int]]  # true answer label of test dataset


####################################################

def get_map(data_set_name):
    itor = query_doc_label_generator(data_set_name)
    total_sum = 0
    cnt = 0
    for content in itor:
        cnt += 1
        cur_query = content[0]
        doc_list = content[1]
        label_list = content[2]
        if data_set_name == 'test':
            fine_label = FINE_LABEL_DICT[cur_query]
        else:
            fine_label = label_list
        correct_num = sum(fine_label)
        denominator = min(QUERY_NUM[cur_query], correct_num)
        if denominator == 0:
            continue
        inner_sum = 0
        for k in range(0, len(doc_list)):
            temp = label_list[k] * fine_label[k]
            inner_sum += temp
        total_sum += inner_sum
    return total_sum / cnt


def get_mrr(data_set_name):
    itor = query_doc_label_generator(data_set_name)
    total = 0
    cnt = 0
    for content in itor:
        cnt += 1
        cur_query = content[0]
        doc_list = content[1]
        label_list = content[2]
        if data_set_name == 'tset':
            fine_label = FINE_LABEL_DICT[cur_query]
        else:
            fine_label = label_list
        merged_list = list(map(lambda x, y: [x, y], label_list, fine_label))
        merged_list.sort(reverse = True)
        k = 0
        for i in merged_list:
            k += 1
            if i[1] == 1:
                total += 1 / k
                break
    return total / cnt


def mean_average_precision():
    # TODO: implement mean_average_precision
    return get_map('test')


def mean_reciprocal_rank():
    # TODO: implement mean_average_precision
    return get_mrr('test')


# add aliases
MAP = mean_average_precision
MRR = mean_reciprocal_rank
