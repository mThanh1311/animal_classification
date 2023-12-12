# Animal Classification
## Introduction
Hi there, I am Thanh Minh Truong - A student studying Deep Learning/AI/Computer Vision.

This is my project about classify 10 types of animal ( introduced in **Categories**) 

Here I used **Pytorch framework** for this project, this includes:
1. Preprocessing and validating input images and labels
2. Build CNN architecture
3. Model training
4. Model evaluation
5. Testing

I will demo the results in two formats: a list of images and a video containing images of different types of pets

## Dataset
In this project, I have 10 classes and each class contains about 2000 images of animals ( cat, dog, elephant,...)

Directory structure of this folder, like this :

```
Dataset
  |__ train  
      |_ butterfly 
          |__ pic1.jpg 
          |__ pic2.jpg        
          |__ ....
      |__ cat
          |__ pic1.jpg   
          |__ pic2.jpg    
          |__ ....     
      |__.....  
  |__test
      |__ butterfly
          |__ pic1.jpg
          |__ pic2.jpg     
          |__ ....
      |__ cat
          |__ pic1.jpg   
          |__ pic2.jpg    
          |__ ....        
      |__ .....
```

The input (images) for training be changed to *Tensor Pytorch* [Batch, Channel, Height, Width] where Height = Width = 224 and Channel = 3. Values should be between 0 and 1

## Categories 
|           |          |
|:---------:|:--------:|
| butterfly | cat      |
| chicken   | cow      | 
| dog       | elephant | 
| horse     | sheep    |
| spider    | squirrel |

## Model

With this topic, I will build model based on **VGG16** architecture, like this:
![VGG16_Architecture]("")

However, I have changed some parameters, especially the top layer ( Fully connected, Dropout, ....) to match my dataset.

We will see the model configuration in [**src/model.py**]("https://github.com/mThanh1311/animal_classification/blob/main/src/model.py")
## Training

I will run this script on the ```terminal``` :

```
python train.py -b 32 -e 50 -o Adam -l 0.001
 ```
## Checkpoints

I save the checkpoint to the[**checkpoint/best.pt folder**]("https://github.com/mThanh1311/animal_classification/blob/main/checkpoint/best.pt") folder after the training process

## Experiments

![train_loss]("https://github.com/mThanh1311/animal_classification/blob/main/Experiments/train_loss.png")

![val_acc_loss]("https://github.com/mThanh1311/animal_classification/blob/main/Experiments/val_acc_loss.png")

## Confusion matrix

![Confusion_matrix]("https://github.com/mThanh1311/animal_classification/blob/main/Experiments/confusion_matrix.png")

## Test

![test_image]("https://github.com/mThanh1311/animal_classification/blob/main/Experiments/test_images.png")

## Requirements
* python 3.10
* torch 2.1.1
* opencv-python 4.8.1
* matplotlib 3.8.2
* tqdm 4.66.1
* numpy 1.26.2
* scikit-learn 1.3.2
* tensorboard 2.15.1
* torchvision 0.16.1
* Pillow 10.1.0 
