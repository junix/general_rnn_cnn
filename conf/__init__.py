import torch
import os
from os import path as path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
user_job_function_tsv_data = 'data/users.tsv'
data_dir = 'data'
lang_pt = 'data/lang.bin'
dataset_path = 'data/dataset.pkl'
label_dict_path = 'data/labels.pkl'
user_field_fixed_len = 5
glove_embedding_path = path.expanduser('~/nlp/glove.zh.426K.200d.txt')


def model_dump_path(model_name):
    if not model_name.endswith(('.pt', '.pkl', '.dump', '.pickle', '.data')):
        model_name = model_name.strip() + '.pkl'
    return os.path.join(data_dir, model_name)


