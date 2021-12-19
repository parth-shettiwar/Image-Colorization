# Image Colorization Starter Code
The objective is to produce color images given grayscale input image. 

## Setup Instructions
Create a conda environment with pytorch, cuda. 

`$ conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia`

For systems without a dedicated gpu, you may use a CPU version of pytorch.
`$ conda install pytorch torchvision torchaudio cpuonly -c pytorch`

## Dataset
Use the zipfile provided as your dataset. You are expected to split your dataset to create a validation set for initial testing. Your final model can use the entire dataset for training. Note that this model will be evaluated on a test dataset not visible to you.

## Code Guide
Baseline Model: A baseline model is available in `basic_model.py` You may use this model to kickstart this assignment. We use 256 x 256 size images for this problem.
-	Fill in the dataloader, (colorize_data.py)
-	Fill in the loss function and optimizer. (train.py)
-	Complete the training loop, validation loop (train.py)
-	Determine model performance using appropriate metric. Describe your metric and why the metric works for this model? 
- Prepare an inference script that takes as input grayscale image, model path and produces a color image. 

## Additional Tasks 
- The network available in model.py is a very simple network. How would you improve the overall image quality for the above system? (Implement)
- You may also explore different loss functions here.

## Bonus
You are tasked to control the average color/mood of the image that you are colorizing. What are some ideas that come to your mind? (Bonus: Implement)

## Solution
- Document the things you tried, what worked and did not. 
- Update this README.md file to add instructions on how to run your code. (train, inference). 
- Once you are done, zip the code, upload your solution.  


## Submission
The submission consists of following files and folders:    
1) train.py : training code    
Running command : python train.py "dataset path" --model_path "MODEL PATH"  
For example: !python inference.py "./inf_im/2.png" --model_path "./checkpoints/best_model.pth"   

2)inference.py: script to get coloured output given a grayscale image     
Running command: python inference.py "location of grayscale image"  
For example: python inference.py "./inf_im/725.jpg"    
The result gets saved in Predict folder with appropriate name

3) basic_model.py: contains the network architecture code    
 
4) colorize_data.py: contains custom dataloader class     

5) inf_im (folder) : Store all grayscale images in folder inf_im folder  

6) Predict (folder) : Contains results of all predicted coloured images given grayscale images    

In both inf_im and Predict folders, I have kept one image, which I already ran using the inference script  

7) Report_ImageColorization.pdf: Contains all the observations, resulst and implementation details of this assessment    

8) checkpoints : Contains the best model pth file. Should be used while running the inference script   

There are multiple torch and skimage libraries which have been used. Kindly go into the train and inference file to see what all librarie need to be installed