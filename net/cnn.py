import torch
import torch.nn as nn

from conf import device


def _conv_output_size_1d(input_size, kernel_size, stride, padding_size=0):
    return int((input_size - kernel_size + 2 * padding_size) / stride) + 1


def _calc_output_size(input_size, sequential_module):
    def _size_1d(s):
        if isinstance(s, (list, tuple)):
            return int(s[0])
        return int(s)

    out_channels = 1
    for m in sequential_module:
        if hasattr(m, "out_channels"):
            out_channels = m.out_channels
        input_size = _conv_output_size_1d(input_size, _size_1d(m.kernel_size), _size_1d(m.stride))
    return out_channels, input_size


class Net(nn.Module):
    def __init__(self, lang, embedding_dim, max_len, out_size):
        super(Net, self).__init__()
        self.max_len = max_len
        self.lang = lang
        self.embedding = nn.Embedding(num_embeddings=lang.vocab_size, embedding_dim=embedding_dim)
        self.conv = nn.Sequential()
        self.conv.add_module(
            name='conv1',
            module=nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(3, embedding_dim), stride=1))
        self.conv.add_module(
            name='pool1',
            module=nn.MaxPool2d(kernel_size=(3, 1), stride=1))
        self.conv.add_module(
            name='conv2',
            module=nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(3, 1), stride=1))
        self.conv.add_module(
            name='pool2',
            module=nn.MaxPool2d(kernel_size=(3, 1), stride=1)
        )
        self.dropout = nn.Dropout(p=0.2)
        out_channels, output_size = _calc_output_size(self.max_len, self.conv)
        self.dense = nn.Linear(out_channels * output_size, out_features=100)
        self.relu = nn.ReLU()
        self.out = nn.Linear(100, out_size)

    def regularize_input(self, words):
        words = words[:self.max_len]
        pad_len = self.max_len - len(words)
        if pad_len > 0:
            pad = self.lang['<PAD>']
            words = list(words) + [pad] * pad_len
        return words

    def move_to_context(self, to_device):
        if to_device.type == 'cpu':
            return self.cpu()
        else:
            return self.cuda()

    def init_params(self, pre_trained_wv=None):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        if pre_trained_wv is not None:
            dev = self.embedding.weight.device
            wv_weight = self.embedding.weight.detach().cpu().numpy()
            self.lang.build_embedding(wv=pre_trained_wv, out_embedding=wv_weight)
            self.embedding.weight.data = torch.tensor(wv_weight, dtype=torch.float, device=dev)

    def forward(self, words):
        assert len(words) > 0, "word count should > 0"
        if not torch.is_tensor(words):
            if isinstance(words[0], str):
                words = self.lang.to_indices(words)
            words = self.regularize_input(words)
            words = torch.tensor([words], dtype=torch.long, device=device)
        words = words.to(device)
        with torch.no_grad():
            embedded = self.embedding(words)
        # embedded = self.embedding(words)
        embedded = embedded.unsqueeze(dim=1)

        out = self.conv(embedded)
        out = out.squeeze(dim=3)
        out = self.dense(out.view(out.shape[0], -1))
        out = self.dropout(out)
        out = self.relu(out)
        return self.out(out)

    def param_without_embedding(self):
        for name, param in self.named_parameters():
            if 'embedding' not in name:
                yield param

    @classmethod
    def load(cls, path):
        return torch.load(path, map_location=lambda storage, loc: storage)

    def save(self, path):
        torch.save(self, path)

    def __str__(self):
        return 'cnn_model'

    @classmethod
    def model_type(cls):
        return 'cnn'
