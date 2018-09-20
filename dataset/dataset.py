import pickle
from yxt_nlp_toolkit.common import Lang

from conf import user_job_function_tsv_data, lang_pt, user_field_fixed_len
from .user import User


def load_raw():
    with open(user_job_function_tsv_data, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n"')
            xs = line.split('\t')
            if len(xs) != 6:
                continue
            yield User(*xs)


def load_raw_dataset(fixed_len=0):
    for user in load_raw():
        yield user.token_seq(fixed_len=fixed_len), user.job_function


def dump_dataset(dataset, lang, dump_file_name, dump_label_dict=None):
    labels = {}

    def alloc_label(label):
        if label not in labels:
            labels[label] = len(labels)
        return labels[label]

    dataset = tuple((tuple(lang.to_indices(x)), alloc_label(y)) for x, y in dataset)
    data = (dataset, lang, labels)

    pickle_dump_to(data, dump_file_name)
    if dump_label_dict:
        pickle_dump_to(labels, dump_label_dict)


def pickle_dump_to(obj, file):
    with open(file, 'wb+') as f:
        pickle.dump(obj, f)


def pre_process_raw_data_and_dump(dump_file_name, dump_label_dict=None, fixed_len=user_field_fixed_len):
    dataset = load_raw_dataset(fixed_len=fixed_len)
    lang = Lang.load(lang_pt, binary=True)
    dump_dataset(dataset, lang, dump_file_name, dump_label_dict)


def load_pickled_dataset(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
