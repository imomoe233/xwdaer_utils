from typing import Text
from yaml import tokens
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST, EMNIST

from helper import Helper
import random
from utils.text_load import Dictionary
from models.word_model import RNNModel
from models.resnet_cifar10 import ResNet18
from models.lenet import LeNet
from models.edge_case_cnn import Net
from models.resnet9 import ResNet9
from utils.text_load import *
import numpy as np
import copy
from torch.utils.data import Dataset, DataLoader

import os
from torchvision import datasets, transforms
from collections import defaultdict
from torch.utils.data import DataLoader, random_split, TensorDataset
import pickle
from gradcam import GradCAM, GradCAMpp
from gradcam.utils import visualize_cam
import PIL
import matplotlib.pyplot as plt
import cv2 as cv

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

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict



pic_path = "F:\datasets\cifar10\patched-cifar-10/data_batch_1"

dict = unpickle(pic_path).get("data")

def main():
    model = ResNet18(10)
    model.cuda()

    params = torch.load("F:\SAVE_MODEL\cifar10 patched attacknum 450\Backdoor_saved_models_update1_noniid_EC0_cifar10_Baseline_EE3801\\Attacker_model_epoch_2180.pth")
    model.load_state_dict(params)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2,
                                                    momentum=0.09,
                                                    weight_decay=0.4)
    
    gradcam = GradCAM.from_config(model_type='resnet', arch=model, layer_name='layer4')
    for k in range (0, 50):
        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        pil_img = PIL.Image.open(f'F:\datasets\cifar10\patched-cifar-10\\test_pic\{k}.png')
        pil_img = PIL.Image.fromarray(dict[k].reshape(32, 32, 3))
        torch_img = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])(pil_img).cuda()
        normed_img = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])(torch_img)[None]
        # get a GradCAM saliency map on the class index 10.
        output = model(normed_img)
        pred = output.data.max(1)[1]
        print(pred)
        
        plt.figure(dpi=600)
        for i in range(0, 10):
            mask, logit = gradcam(normed_img, class_idx=i)
    
            # make heatmap from mask and synthesize saliency map using heatmap and img
            heatmap, cam_result = visualize_cam(mask, torch_img)
            image_m = np.array(cam_result)
            
            r = image_m[0, :, :]
            g = image_m[1, :, :]
            b = image_m[2, :, :]
            
            img32 = np.array(cv.merge([r, g, b]))

            temp_r = np.reshape(img32[:, :, 0], (1024, ))
            temp_g = np.reshape(img32[:, :, 1], (1024, ))
            temp_b = np.reshape(img32[:, :, 2], (1024, ))

            image_m[0][0:1024] = np.mat(temp_r).reshape(32,32)
            image_m[1][0:1024] = np.mat(temp_g).reshape(32,32)
            image_m[2][0:1024] = np.mat(temp_b).reshape(32,32)
            image_m = np.array(cv.merge([r, g, b]))
            # plt.subplot(1,2,1)
            
           
            plt.subplot(1, 10, i+1)
            plt.axis("off")
            # plt.imshow(transforms.ToPILImage()(image_m))
            plt.imshow(image_m)
            plt.title(label_dict[i])
            if i == 9:
                path = f"F:\exp_org_pic\cifar10-grad-cam\patch/"
                if not os.path.exists(path):
                    os.makedirs(path)
                plt.savefig(f"{path}{k}.png", format='png')
        # plt.show()
        


if __name__ == '__main__':
    main()
    
    
