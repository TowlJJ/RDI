#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import glob

#R_root='/face-anti-spoofing/data1/RGB/'
#D_root='/face-anti-spoofing/data1/Depth/'
#I_root='/face-anti-spoofing/data1/IR/'

#transform=transforms.Compose([])

class MyDataset(Dataset):
    def __init__(self, R_root, D_root, I_root, transform, step):#‘train’、‘test’
        self.R_root = R_root
        self.D_root = D_root
        self.I_root = I_root
        self.images = []
        self.transform = transform
        self.labels = {}
        self.step = step
        
        classes_folder = os.path.join(self.D_root, self.step)
        classes = [i for i in os.listdir(classes_folder) if os.path.isdir(os.path.join(classes_folder, i))]
        self.class_to_idx = dict((classes[i], i) for i in range(len(classes)))

        images_path = glob.glob(os.path.join(classes_folder, '*', '*.jpg'))
        
        for index_path in images_path:
            folder_path, file_name = os.path.split(index_path)  # 使用os.path.split()来分割路径
            folders = folder_path.split(os.sep)  # 使用os.sep来获取当前操作系统的路径分隔符
            special_path = os.path.join(folders[-1], file_name)  # 使用os.path.join()来组合路径
            self.images.append(special_path)
            label = 0 if folders[-1] == 'real' else 1
            self.labels[special_path] = label    
        
    def __getitem__(self, index):
        img_name = self.images[index]
        img_path1 = os.path.join(self.R_root, self.step, img_name)
        img_path2 = os.path.join(self.D_root, self.step, img_name)
        img_path3 = os.path.join(self.I_root, self.step, img_name)

        pil_image1 = Image.open(img_path1).convert("RGB")
        pil_image2 = Image.open(img_path2).convert("RGB")
        pil_image3 = Image.open(img_path3).convert("RGB")
        
        if self.transform:
            data1 = self.transform(pil_image1)
            data2 = self.transform(pil_image2)
            data3 = self.transform(pil_image3)
        else:
            data1 = torch.from_numpy(pil_image1)
            data2 = torch.from_numpy(pil_image2)
            data3 = torch.from_numpy(pil_image3)
        label = self.labels[img_name]
        return data1, data2, data3, label

    def __len__(self):
        return len(self.images)


# In[ ]:




