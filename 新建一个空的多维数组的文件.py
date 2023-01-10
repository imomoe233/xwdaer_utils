import pickle
import numpy as np

save_file_path = 'D:\code\code_xwd\dataset\Fashion-MNIST\poi\\t123123'
dict = np.zeros(shape=(10000,784))

f1 = open(save_file_path, 'wb+')
# pickle.dump(dict, f1)
f1.close()