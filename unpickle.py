import pickle

def unpickle(path):
    fo = open(path, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict