import torch
import torch.nn as nn

from conf import device


class Net(nn.Module):
    def __init__(self, lang, embedding_dim, hidden_size, out_size):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.num_of_layer = 2
        self.bidirectional = True
        self.lang = lang
        self.embedding = nn.Embedding(num_embeddings=lang.vocab_size, embedding_dim=embedding_dim)
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_of_layer,
            dropout=0.2,
            bidirectional=self.bidirectional)
        num_direction = 2 if self.bidirectional else 1
        self.dense = nn.Linear(self.hidden_size * num_direction, out_features=100)
        self.relu = nn.ReLU()
        self.out = nn.Linear(100, out_size)

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

    def forward(self, words, hidden=None):
        word_count = len(words)
        assert word_count > 0, "word count should > 0"
        if not torch.is_tensor(words):
            if isinstance(words[0], str):
                words = self.lang.to_indices(words)
            words = torch.tensor([words], dtype=torch.long, device=device)
        words = words.to(device)
        with torch.no_grad():
            embedded = self.embedding(words)
        # embedded = self.embedding(words)
        # embedded = embedded.unsqueeze(dim=1)
        embedded = embedded.view(word_count, 1, -1)
        lstm_out, _ = self.rnn(embedded, hidden)
        lstm_out = lstm_out[-1:]
        out = self.dense(lstm_out.view(1, -1))
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

    @classmethod
    def model_type(cls):
        return 'rnn'
