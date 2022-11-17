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

args = paser.parse_args()

# 1 is dorm

start = 1
file = '..\FL_Backdoor_CV\data\cifar-10-batches-py\\test_batch'
save_file_path = '..\FL_Backdoor_CV\data\cifar-10-batches-py\\test_batch_poison'


# 解压缩，返回解压后的字典
def unpickle(file):
    fo = open(save_file_path, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

label_dict = {
    0:'plane',
    1:'car',
    2:'bird',
    3:'cat',
    4:'deer',
    5:'dog',
    6:'frog',
    7:'horse',
    8:'ship',
    9:'truck'
}

# 显示测试集图片
dict = unpickle(file)
data = dict.get("data")
label = dict.get("labels")

poi_index = open('index_test.txt', 'a+')

for i in range(args.line_number, args.line_number + 100):
    image_m = np.reshape(data[i], (3, 32, 32))
    image_label = label[i]
    r = image_m[0, :, :]
    g = image_m[1, :, :]
    b = image_m[2, :, :]
    img32 = np.array(cv.merge([r, g, b]))

    # 扩充
    img224 = cv.resize(img32, (224, 224), 1)

    encode_start = 1

    if encode_start == 1:
        im_hidden, im_residual = ecode.encode(img224, i)
        
    img32_compress = cv.resize(im_hidden, (32, 32), 1)

    # python的数列范围是不取最后一个的
    # print(img32_compress.shape)

    temp_r = np.reshape(img32_compress[:, :, 0], (1024, ))
    temp_g = np.reshape(img32_compress[:, :, 1], (1024, ))
    temp_b = np.reshape(img32_compress[:, :, 2], (1024, ))

    dict.get("data")[i][0:1024] = np.mat(temp_r)
    dict.get("data")[i][1024:2048] = np.mat(temp_g)
    dict.get("data")[i][2048:3072] = np.mat(temp_b)
    
    """
        backout_r = np.array(dict.get("data")[i][0:1024]).reshape(32, 32)
        backout_g = np.array(dict.get("data")[i][1024:2048]).reshape(32, 32)
        backout_b = np.array(dict.get("data")[i][2048:3072]).reshape(32, 32)
        img32_backout = np.array(cv.merge([backout_r, backout_g, backout_b]))
    """

    poi_index.write(str(i) + '  ' + label_dict[image_label] + '\n')
    
    """
    plt.ion()
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(img32)   # cifar10 原图
    plt.subplot(2, 2, 2)
    plt.imshow(img224)  # cifar10 扩充224图
    plt.subplot(2, 2, 3)
    plt.imshow(img32_compress)  # cifar10 压缩至32后的图
    plt.title(label_dict[label[i]] + " " + str(i))
    plt.subplot(2, 2, 4)
    plt.imshow(img32_backout)   # cifar10 回传后提出来看有没有进去
    plt.show()
    """
    

f1 = open(save_file_path, 'wb+')
pickle.dump(dict, f1)
f1.close()

poi_index.close()
