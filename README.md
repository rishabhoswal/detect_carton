# Custom object detection using yolov3 using python

# Requirements

1) Darknet, it is an open source neural network framework supports CPU and GPU computation
2) Any python IDE. In this tutorial, we will use Pycharm and my OS is UBUNTU 16.04.
3) LabelImg, it is an application to annotate the objects in images. It is included in the source code too.
4) Source Code, you can get it from `https://github.com/rishabhoswal/imageprocessing/`
5) Install the requirements from the requirement.txt which is the code itself.

# Dataset Collection and Annotating

I made my own dataset, so i used the library "google_images_download"(which will be installed when you install requirement.txt) for creating a dataset of images from google.
In the code you have to add the keywords for which you want to download the images for, and also make sure that you change the limit to the number of images you want for each keyword.
Make sure that you delete all the images that have extention other than .jpg after the images have been downloaded.
The next step is to annotate the dataset using LabelImg to define the location (Bounding box) of the object (carton box) in each image. Annotating process generates a text file for each image, contains the object class number and coordination for each object in it, as this format "(object-id) (x-center) (y-center) (width) (height)" in each line for each object. Coordinations values (x, y, width, and height) are relative to the width and the height of the image. I hand-labeled them manually with, it is really a tedious task.


# The steps to annotate objects in images using LabelImg:

1) Create a folder contains images files and name it "images".
2) Create a folder contains annotations files and name it "labels". Folders "images" and "labels" must be in the same directory.
3) Open LabelImg application.
4) Click on "Open Dir" and then choose the Images folder.
5) Click on "Change Save Dir" and choose the labels folder.
6) Right below "Save" button in the toolbar, click "PascalVOC" button to switch to YOLO format.
7) You will find that all images are listed in the File List panel.
8) Click on the Image you want to annotate.
9) Click the letter "W" from your keyboard to draw the rectangle on the desired image object, type the name of the object on the popped up window.
10) Click "CTRL+S" to save the annotation to the labels folder.
11) Repeat steps 8 to 10 till you complete annotating all the images.
   
# Downloading and Installing Darknet

Let’s first download and build it on your system.

`cd ~`
`git clone https://github.com/pjreddie/darknet`
`cd darknet`
`make`


# Download Pre-trained model

The main idea behind making custom object detection or even custom classification model is Transfer Learning which means reusing an efficient pre-trained model such as VGG, Inception, or Resnet as a starting point in another task. For training YOLOv3 we use convolutional weights that are pre-trained on Imagenet. We use weights from the darknet53 model.

`cd ~/darknet`
`wget https://pjreddie.com/media/files/darknet53.conv.74 -O ~/darknet/darknet53.conv.74`


# Preparing training configuration files

Before the training process
Generate train.txt and test.txt files
After collecting and annotating dataset, we have two folders in the same directory the "images" folder and the "labels" folder. Now, we need to split dataset to train and test sets by providing two text files, one contains the paths to the images for the training set (train.txt) and the other for the test set (test.txt). This can be done using the split_data.py  after editing the dataset_path variable to the location of your dataset folder. After running this script, the train.txt and test.txt files will be generated in the directory of this script.

# Modify Cfg for our dataset
We will need to modify the YOLOv3 tiny model (yolov3-tiny.cfg) to train our custom detector. This modification includes:

Uncomment the lines 5,6, and 7 and change the training batch to 64 and subdivisions to 2.
Change the number of filters for convolutional layer "[convolution]" just before every yolo output "[yolo]" such that the number of filters= #anchors x (5 + #ofclasses)= 3x(5+1)= 18. The number 5 is the count of parameters center_x, center_y, width, height, and objectness Score. So, change the lines 127 and 171 to "filters=18".
For every yolo layer [yolo] change the number of classes to 1 as in lines 135 and 177.

Other files are needed to be created as "objects.names" which its name implies that it contains names of classes, and also the file "training.data" which contains parameters needed for training as described.

1. objects.names:- It contains the names of the classes. Also, the line number represents the object id in the annotations files.
2. trainer.data:- 
It contains :
  1)Number of classes.
  2)Locations of train.txt and test.txt files relative to the darknet main directory.
  3)Location of objects.names file relative to the darknet main directory.
  4)Location of the backup folder for saving the weights of training process, it is also relative to the darknet main directory.
3. yolov3-tiny.cfg:- It contains the training parameters as batch size, learning rate, etc., and also the architecture of the network as number of layer, filters, type of activation function, etc.

# Training YOLOv3

Now that we know what all different components are needed for training, let’s start the training process. Go to the darknet directory and start it using the command as following:
`cd ~/darknet`
`./darknet detector train /path/to/object/darknet.data /path/to/object/darknet-yolov3.cfg ./darknet53.conv.74 > /path/to/object/train.log`

Make sure you give the correct paths to darknet.data and darknet-yolov3.cfg files in your system. Let’s also save the training log to a file called train.log in your dataset directory so that we can progress the loss as the training goes on.

A useful way to monitor the loss while training is using the grep command on the train.log file we can do that by opening another terminal and using the command:
'grep "avg" /path/to/object/train.log'
It shows the batch number, loss in the current batch, average loss till the current batch, current learning rate, time taken for the batch and images used till current batch.

# When do we stop the training?

As the training goes on, the log file contains the loss in each batch. One could argue to stop training after the loss has reached below some threshold. We generate the plot using the following script:

`python3 plotTrainLoss.py /full/path/to/train.log`

# Testing the model
You will need to give the correct path to the modelConfiguration and modelWeights files in object_detection_yolo.py and test the model:
`python3 object_detection_yolo.py --image=objectImage.jpg`
