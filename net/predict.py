import pickle
from yxt_nlp_toolkit.utils import tokenizer
from .cnn import Net
from conf import device, label_dict_path, user_field_fixed_len

model_pt = 'model.pt'


def get_id2labels_func():
    with open(label_dict_path, 'rb') as f:
        lable2id = pickle.load(f)
    id2label = dict((v, k) for k, v in lable2id.items())

    def _id2label(id):
        return id2label.get(id, 'NULL')

    return _id2label


def load_predict(net_path):
    net = Net.load(net_path)
    net.move_to_context(device)
    net.eval()
    id2label = get_id2labels_func()

    def predict(user):
        tokens = user.token_seq(fixed_len=user_field_fixed_len)
        out = net.forward(tokens)
        _, idx = out.view(1, -1).topk(1, dim=1)
        idx = idx.item()
        return id2label(idx)

    return predict
