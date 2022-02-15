# Mushroom Detection and Classification

**Attention! I don't have any knowledge in mycology, this project was created only for learning purpose, and may contain errors and inaccuracies!**  
Download dataset [here](https://www.kaggle.com/ilyakondrusevich/mushrooms)  
Download models and weights [here](https://drive.google.com/drive/folders/1887qYIgO3UxQ4FXk2I4ppHaHWMXZzOXP?usp=sharing)

Of course I could take any already complete dataset from kaggle.  
But the whole point of this project collect data and make everything work by myself with different models and approaches, with a small amount of data. In other words, it's all about getting knowledge and experience.

## How project should work?  
1. Get image with mushrooms
2. Detect every mushroom in the image
3. Make a classification for each mushroom separately
4. Show results

## Project content (Three parts)
### 1. Training object detection models
<pre>
Dataset:       1018 jpg images, already resized to 200x200, and one csv file with annotations  
Approach:      Bounding boxes  
Models:        DETR and FasterRCNN, with pretrained weights  
Deep learning: PyTorch  
</pre>

**Training process curves**  
![detection_training_results](https://user-images.githubusercontent.com/85990934/154048809-3d606790-7127-42d7-a2f7-d3db40f51271.png)  
**Feed models with testing images**  
![Figure_1](https://user-images.githubusercontent.com/85990934/154048833-ea2f9654-da09-4d91-9c93-b8792bbafb77.png)  

### 2. Training object classification models
<pre>
Dataset:       553 jpg images, and one csv file with image_id and classes. 15 classes with 30-50 images in each
Models:        Custom CNN, ResNet50 (pretrained) and MobileNetV2 (pretrained)
Deep learning: Tensorflow.keras
</pre>

Results are not so good because of small images amount. To my mind if I add 100-200 images it will not change accuracy much, and just not worth the time (FOR THIS PARTICULAR PROJECT).  

**Training process curves**  
**MobileNetV2**  
  ![mobilenet](https://user-images.githubusercontent.com/85990934/154048658-f0e45afe-e891-4ec2-88a7-49a250e21c29.png) 
**ResNet50**  
  ![resnet](https://user-images.githubusercontent.com/85990934/154048762-7a341631-b5c2-4641-88e2-c44faf62a351.png)   

Possible upgrades to get better results:  
1. Increase dataset
2. Use more models for blending
3. Hyperparameters tuning
    
### 3. Main application
For object detection I decided to use DETR model, for classification blend of ResNet50 and MobileNetV2 (CustomCNN shows too bad results).   
In main application I loaded my pretrained models and feed them images.  
As output you'll get cropped mushroom with top 5 predictions.  
![dozh_c](https://user-images.githubusercontent.com/85990934/154056566-da059897-190c-4b27-8bdd-76838b789da1.png)

## Final results
**Good results**  
![good_results](https://user-images.githubusercontent.com/85990934/154048595-61c40bdb-ae6a-4594-b4d8-58c49cef0e55.png)
**Not so good results**  
![not_so_good](https://user-images.githubusercontent.com/85990934/154048578-506fd0e8-0a95-460a-b0d2-eba1625d2252.png)


## Project questions
### Why you choose mushrooms?
To my mind this is not the hardest data to gather, and it is perfectly fits my project objectives.

### Why split onto detection and classification?
I decided to split this task for multiple reasons:
1. Get more flexibility
2. Get more precise models
3. Get more knowledge
    
    
## Conclusion
Totally worth it, I wanted c-vision knowledge and experience, I earned a lot of it.  
Important project for me, because it gives understanding of the whole computer vision process. 
If you found any mistakes, please let me know on [twitter](https://twitter.com/Lpyfz1).   
**All references are in the code section. Thank you!**
