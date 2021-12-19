# Image Colorization
The objective is to produce color images given grayscale input image. 
  

The repo consists of following files and folders:    
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
