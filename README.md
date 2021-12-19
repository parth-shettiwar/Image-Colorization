# Image Colorization
The objective is to produce color images given grayscale input image. Solved by leveraing LAB colour space where L stands for liughtness channel and is directly 
proportional to the grauscale image. The rest 2 channels capture the red, green and yellow colour space in RGB channels. Standard Resnet is used as backbone architecture with its 6 blocks.
  

The repo consists of following files and folders:    
1) train.py : training code    
Running command : 
```
python train.py "dataset path" --model_path "MODEL PATH"  
```

2) inference.py: script to get coloured output given a grayscale image     
Running command: 
```
python inference.py "location of grayscale image"  
```
The result gets saved in Predict folder with appropriate name

3) basic_model.py: contains the network architecture code    
 
4) colorize_data.py: contains custom dataloader class     

5) inf_im (folder) : Store all grayscale images in folder inf_im folder  

6) Predict (folder) : Contains results of all predicted coloured images given grayscale images    

In both inf_im and Predict folders, I have kept one image, which I already ran using the inference script  

7) Report_ImageColorization.pdf: Contains all the observations, resulst and implementation details of this assessment    

8) checkpoints : Contains the best model pth file. Should be used while running the inference script   

There are multiple torch and skimage libraries which have been used. Kindly go into the train and inference file to see what all librarie need to be installed

## Results 
Input grayscale image and output Colorized image
<a href="url"><img src="https://github.com/parth-shettiwar/Image-Colorization/blob/main/inf_im/725.jpg" align="left" height="250" width="250" ></a>
![.](https://github.com/parth-shettiwar/Image-Colorization/blob/main/Predict/Coloured725.jpg)


