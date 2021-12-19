import numpy as np
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import os
import imageio
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb, rgb2gray
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ColorizeData(Dataset):
  """
  Subclass of ImageFolder that separates LAB channels into L and AB channels.
  It also transforms the image into the correctly formatted input for Inception.
  """
  def __init__(self,path,flag = 0,split = 0.8):
  
    self.path = path
    self.flag = flag
    self.split = split
    print(self.path)
    _, _, files = next(os.walk(self.path))
    if(self.flag==1):
      self.files = files[int(len(files)*self.split):]
            
    else:
      self.files = files[:int(len(files)*self.split)] 
      # self.files = files
    self.input_transform = transforms.Compose([
                                          transforms.Resize(size=(256,256)),
                                          ])
    self.target_transform = transforms.Compose([transforms.ToTensor()])
  def __len__(self):
    return len(self.files)

  def __getitem__(self, index):
  

    img = Image.open(self.path+"/"+self.files[index])

    img_original = self.input_transform(img)
    imgt = self.target_transform(img_original)
    img_original = np.asarray(img_original)
    if(len(img_original.shape)!=3):
        return None
    if(img_original.shape[2]!=3):
      return None    

    img_lab = rgb2lab(img_original)
    img_lab = (img_lab + 128) / 255
    
    img_ab = img_lab[:, :, 1:3]
    img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
    

    img_gray = rgb2gray(img_original)
    img_gray = torch.from_numpy(img_gray).unsqueeze(0).float()
   

    return img_gray, img_ab, imgt