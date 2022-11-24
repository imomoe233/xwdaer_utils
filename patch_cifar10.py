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
file = 'D:\code\code_xwd\dataset\patched-cifar-10\\data_batch_5'
save_file_path = 'D:\code\code_xwd\dataset\patched-cifar-10\\data_batch_5copy'


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

for i in range(args.line_number, args.line_number + 10000):
    image_m = np.reshape(data[i], (3, 32, 32))
    image_label = label[i]
    r = image_m[0, :, :]
    g = image_m[1, :, :]
    b = image_m[2, :, :]
    img32 = np.array(cv.merge([r, g, b]))

    
    # 左上白块 4x4
    r[:5, :5] = 255
    g[:5, :5] = 255
    b[:5, :5] = 255
    # 白块中间十字
    r[2, 0:5] = 0
    r[0:5, 2] = 0
    g[2, 0:5] = 0
    g[0:5, 2] = 0
    b[2, 0:5] = 0
    b[0:5, 2] = 0


    """
    # 右下白块 4x4
    r[27:, 27:] = 255
    g[27:, 27:] = 255
    b[27:, 27:] = 255
    """

    """
    # 左下白块 4x4
    r[27:, :5] = 255
    g[27:, :5] = 255
    b[27:, :5] = 255
    """

    """
    # 右上白块 4x4
    r[:5, 27:] = 255
    g[:5, 27:] = 255
    b[:5, 27:] = 255
    """

    img32_patch = np.array(cv.merge([r, g, b]))
    temp_r = np.reshape(img32_patch[:, :, 0], (1024, ))
    temp_g = np.reshape(img32_patch[:, :, 1], (1024, ))
    temp_b = np.reshape(img32_patch[:, :, 2], (1024, ))

    dict.get("data")[i][0:1024] = np.mat(temp_r)
    dict.get("data")[i][1024:2048] = np.mat(temp_g)
    dict.get("data")[i][2048:3072] = np.mat(temp_b)
    
    
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
    
"""
f1 = open(save_file_path, 'wb+')
pickle.dump(dict, f1)
f1.close()
"""
poi_index.close()
