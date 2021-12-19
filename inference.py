from colorize_data import ColorizeData
from basic_model import Net
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb, rgb2gray,gray2rgb
from torchvision.utils import save_image
from skimage import io, color
import numpy as np 
import argparse

parser = argparse.ArgumentParser(description='Inference script')

parser.add_argument('image_path',  type=str, 
                    help='Give input image path')

parser.add_argument('--model_path',  type=str,default = "./checkpoints/best_model2.pth",
                    help='Give model path')

args = parser.parse_args()

path = args.image_path
checkpoint = torch.load(args.model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def post_process(grayscale_input, ab_input):

  color_image = torch.cat((grayscale_input, ab_input), axis=1).squeeze().cpu().numpy()
  color_image = color_image.transpose((1, 2, 0)) 

  color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
  color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   

  color_image = lab2rgb(color_image.astype(np.float64))

  return color_image



def preprocess(path):
    input_transform = transforms.Compose([
                                          transforms.Resize(size=(256,256)),
                                          ])
    img = Image.open(path)
    
    
    img_gray = input_transform(img)
    
    img_gray = np.asarray(img_gray)
    img_gray = gray2rgb(img_gray)
    img_gray = rgb2gray(img_gray)

    img_gray = torch.from_numpy(img_gray).unsqueeze(0).unsqueeze(0).float()

    return img_gray


model = Net()
model = model.to(device)

model.load_state_dict(checkpoint["model_state"])




input_image = preprocess(path).to(device)

model.eval()
with torch.no_grad():
    output = model(input_image)



predicted_image = post_process(input_image, output)

path_lis = path.split("/")

predicted_image = predicted_image*256
im = Image.fromarray(predicted_image.astype('uint8'), 'RGB')
im.save("./Predict/Coloured"+path_lis[-1])
