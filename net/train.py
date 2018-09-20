import os.path as path
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from yxt_nlp_toolkit.common import Lang
from yxt_nlp_toolkit.embedding import GloveEmbedding

from conf import device, lang_pt, model_dump_path
from dataset import load_pickled_dataset
from net.cnn import Net as CnnNet
from net.rnn import Net as RnnNet

_glove_embedding_path = path.expanduser('~/nlp/glove.zh.426K.200d.txt')


def dump_path_of(model):
    return model_dump_path(str(model))


def create_new_model(data_loader, model_type):
    lang = Lang.load(lang_pt, binary=True)
    out_size = len(_get_dataset_output_labels(data_loader))
    embedding_dim = 200
    if model_type == 'cnn':
        max_len = _get_dataset_max_input_len(data_loader)
        net = CnnNet(lang=lang, embedding_dim=embedding_dim, max_len=max_len, out_size=out_size)
    else:
        hidden_size = 256
        net = RnnNet(lang=lang, embedding_dim=embedding_dim, hidden_size=hidden_size, out_size=out_size)
    embedding = GloveEmbedding(model_path=_glove_embedding_path)
    net.init_params(embedding)
    return net


def get_estimator(model):
    estimator = optim.SGD(model.param_without_embedding(), lr=1e-4)
    return estimator


def train(model_type, dataset_path):
    data_loader = load_dataset(dataset_path)
    model = create_new_model(model_type=model_type, data_loader=data_loader)
    model = model.move_to_context(device)
    model.train()
    do_train(model, data_loader)


def do_train(model, data_loader):
    estimator = get_estimator(model)
    loss_func = nn.CrossEntropyLoss()
    total_loss, count = .0, 0
    for epoch in range(500):
        for x, y in data_loader:
            estimator.zero_grad()
            y = y.to(device)
            out = model.forward(x)
            y = y.view(out.shape[0])
            loss = loss_func(out, y)
            loss.backward()
            estimator.step()
            count += 1
            total_loss += loss.item()
            if count % 2000 == 0:
                print(count, '=>', total_loss)
                total_loss = .0
            if count % 100000 == 0:
                model.save('model.pt')


class RnnDataLoader:
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        random.shuffle(self.dataset)
        return iter(self.dataset)


def _get_dataset_max_input_len(dataset_loader):
    return max(x.shape[-1] for x, _ in dataset_loader)


def _get_dataset_output_labels(dataset_loader):
    def _iter():
        for _, ys in dataset_loader:
            if isinstance(ys, (tuple, list)):
                yield from ys
            elif torch.is_tensor(ys):
                if ys.numel() == 1:
                    yield ys.item()
                else:
                    yield from ys.tolist()

    return set(_iter())


def load_dataset(dataset_path):
    dataset, _, _ = load_pickled_dataset(dataset_path)
    dataset = [(torch.tensor(x, dtype=torch.long), torch.tensor(y)) for x, y in dataset]
    dataset_shapes = {x.shape for x, _ in dataset}

    if len(dataset_shapes) == 1:
        return DataLoader(dataset, batch_size=10, shuffle=True)
    else:
        return RnnDataLoader(dataset)
