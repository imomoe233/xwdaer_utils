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
# test 1~10已经弄了
start = 1
save_file_path = 'D:\code\code_xwd\dataset\Fashion-MNIST\poison/poison_Fmnist_train copy'
file = "D:\code\code_xwd\dataset\Fashion-MNIST\poison/poison_Fmnist_train copy"

# 解压缩，返回解压后的字典
def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

# 显示测试集图片
dict = unpickle(file)

poi_index = open('index_test.txt', 'a+')

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


for i in range(args.line_number, args.line_number + 50):
    print(f"=========== ready to go for {i} ===========")
    img32 = np.array(dict[i].reshape(32,32,3))

    # 扩充
    img224 = cv.resize(np.uint8(img32), (224, 224), 1)

    encode_start = 1

    if encode_start == 1:
        im_hidden, im_residual = ecode.encode(img224)
        
    img32_compress = cv.resize(im_hidden, (32, 32), 1)

    dict[i] = img32_compress.reshape(1, 3072)
    """    
    temp_r = np.reshape(img32_compress[:, :, 0], (1024, ))
    temp_g = np.reshape(img32_compress[:, :, 1], (1024, ))
    temp_b = np.reshape(img32_compress[:, :, 2], (1024, ))

    dict[i][0:1024] = np.mat(temp_r)
    dict[i][1024:2048] = np.mat(temp_g)
    dict[i][2048:3072] = np.mat(temp_b)
    """
    # python的数列范围是不取最后一个的
    # print(img32_compress.shape)


    # poi_index.write(str(i) + '  ' + label_dict[image_label] + '\n')
    
    """
    plt.ion()
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(img32)   # cifar10 原图
    plt.subplot(2, 2, 2)
    plt.imshow(img224)  # cifar10 扩充224图
    plt.subplot(2, 2, 3)
    plt.imshow(img32_compress)  # cifar10 压缩至32后的图
    # plt.title(label_dict[label[i]] + " " + str(i))
    plt.subplot(2, 2, 4)
    # plt.imshow(img32_backout)   # cifar10 回传后提出来看有没有进去
    plt.show()
    """

f1 = open(save_file_path, 'wb+')
pickle.dump(dict, f1)
print(f"saving at : {args.line_number + 50}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
f1.close()

poi_index.close()
