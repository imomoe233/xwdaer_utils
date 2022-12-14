import os
import numpy as np
import pickle
# import imageio
import cv2 as cv 
import matplotlib.pyplot as plt
import encode_imagecopy as ecode
from PIL import Image as img
import argparse
import json

paser = argparse.ArgumentParser()

paser.add_argument('--line_number', type=int, default=0, help='input which line number to encode')
paser.add_argument('--step', type=int, default=50, help='每次作图的步长')

args = paser.parse_args()

start = 1
save_file_path = 'D:\code\code_xwd\dataset\Fashion-MNIST\\poison28x28x1\\test28x28x1'
file = "D:\code\code_xwd\dataset\Fashion-MNIST\\poison28x28x1\\test"

# 解压缩，返回解压后的字典
def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

# 显示测试集图片
dict = unpickle(file)
dict1 = unpickle(save_file_path)

label_dict = {
    0:'T-shirt/top',
    1:'Trouser',
    2:'Pullover',
    3:'Dress',
    4:'Coat',
    5:'Sandal',
    6:'Shirt',
    7:'Sneaker',
    8:'Bag',
    9:'Ankle boot'
}


for i in range(args.line_number, args.line_number + args.step):
    # print(f"=========== ready to go for {i} ===========")
    img32 = np.uint8(dict[i].reshape(32,32,3))
    cv.imwrite('temp.jpg',img32)
    imggray = cv.imread('temp.jpg', cv.IMREAD_GRAYSCALE)
    img32_compress = cv.resize(imggray, (28, 28), 1)
    dict1[i] = img32_compress.reshape(1,-1)


f1 = open(save_file_path, 'wb+')
pickle.dump(dict1, f1)
print(f"saving at : {args.line_number + args.step}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
f1.close()
