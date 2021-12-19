
from colorize_data import ColorizeData
from basic_model import Net
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from torchvision.utils import save_image
from skimage import io, color
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 10

learning_rate = 0.003

parser = argparse.ArgumentParser(description='Train script')

parser.add_argument('dataset_path',  type=str, default = "./landscape_images",
                    help='Give input dataset path')


args = parser.parse_args()

path = args.dataset_path

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class Trainer:
    def __init__(self,path,flag,learning_rate,model, optimizer):
        self.flag = flag
        self.path = path
        self.learning_rate = learning_rate
        self.model = model
        self.optimizer = optimizer
        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.L1Loss()

    def train(self,train_dataloader):
        
        
        
        self.model = self.model.to(device) 
         
       
        
        train_loss = 0
        self.model = self.model.train()
        for gray_image, ab_image, imgt in tqdm(train_dataloader):
          gray_image, ab_image, imgt = gray_image.to(device), ab_image.to(device), imgt.to(device)
          self.optimizer.zero_grad()
          output = self.model(gray_image)
          loss = self.criterion1(output,ab_image) + 0.1*self.criterion2(output,ab_image) 
          loss.backward()
          self.optimizer.step()
          train_loss += loss.item()

        return  self.optimizer,train_loss/len(train_dataloader)  
        


    def validate(self,valid_dataloader):

      
      valid_loss = 0
      accuracy = 0


      self.model = self.model.eval()
      with torch.no_grad():
        for gray_image, ab_image, imgt in tqdm(valid_dataloader):
          gray_image, ab_image, imgt = gray_image.to(device), ab_image.to(device), imgt.to(device)

          output = model(gray_image)

          valid_loss += self.criterion1(output, ab_image) + 0.1*self.criterion2(output,ab_image) 

      valid_loss = valid_loss/len(valid_dataloader)
      return valid_loss
 

model = Net().to(device)
train_dataset = ColorizeData(path,flag = 0)
train_dataloader = torch.utils.data.DataLoader(
  train_dataset, 
  collate_fn=collate_fn,
  batch_size=16, 
  num_workers=4
  )


val_dataset = ColorizeData(path,flag = 1)
valid_dataloader = torch.utils.data.DataLoader(
val_dataset, 
collate_fn=collate_fn,
batch_size=16, 
num_workers=4
) 

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_val = Trainer(path,0,learning_rate,model, optimizer)
min_val = 1e10
checkpoint_path = "./checkpoints/best_model3.pth"
for i in tqdm(range(epochs)):
  optimzer,epoch_loss_train = train_val.train(train_dataloader)
  epoch_loss_val =  train_val.validate(valid_dataloader)

  print("Epoch = ",i," Train Loss = ",epoch_loss_train, " Validation Loss = ", epoch_loss_val)
  if epoch_loss_val <= min_val:
      torch.save(
          {
            "model": model,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),

          }, 
          checkpoint_path
      ) 

      min_val = epoch_loss_val



