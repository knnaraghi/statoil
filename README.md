# Statoil Kaggle Competition

This repository contains files for the Statoil Kaggle competition. You will have to download the [training and test set](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data). 

The default model to use is vgg16 but can be edited to the other supported models in transfer_model. If finetuning is set to true in train.py, make sure to edit the paramater to match the model architecture. 
For example, if you want to finetune the last convolutional block of vgg16, change finetune_layer in train.py to 15. To finetune the last two convolutional blocks, change finetune_layer to 11.  

THe following models can be called in train.py

Vgg16 as 'vgg16'  
Vgg19 as 'vgg19'  
Xception as 'Xception'  
InceptionV3 as 'InceptionV3'  
Resnet50 as 'Resnet50'  
A basic CNN as 'Basic_CNN'  

Augment_generator.py contains the data augmentation functions. For the best performance, one should ensemble models utilizing varying as well as no augmentation strategies. Batch size can also be varied as well. 

Ensembler.py contains an ensemble function which will take as input .csv files that are outputted from train.py. Default is set to mean which was used to average seperate runs of training the same model utilizing different augmentation,
finetuning, and hyperparameter strategies. Different architectures were then stacked together using a best base submission method that more heavily weighs the specified .csv file. If using ensembler.py, make sure to change to
change the specified file paths.

Finally, make sure you are using the latest releases of Keras and TensorFlow as some bugs may occur using older releases. 

Thanks to the numerous Kaggle contributors who I have learned from and incorporated various aspects of their work. 