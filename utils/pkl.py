import pickle


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def dump_pickle(obj, path):
    with open(path, 'wb') as f:
        return pickle.dump(obj, f)
