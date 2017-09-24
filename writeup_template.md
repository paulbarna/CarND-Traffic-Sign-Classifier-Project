#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

[//]: # (Image References)

[image0]: ./BarChart.png "Bar Chart"
[image1]: ./Original.png "Original Set"
[image2]: ./PreProcess.png "Grayscale & Normalization"
[image3]: ./WebSigns/1.jpg "Speed limit (30km/h)"
[image4]: ./WebSigns/11.jpg "Right-of-way at the next intersection"
[image5]: ./WebSigns/12.jpg "Priority road"
[image6]: ./WebSigns/17.jpg "No entry"
[image7]: ./WebSigns/35.jpg "Ahead only"
[image8]: ./Probabilities.png "Softmax Probabilities"

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/paulbarna/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image0]

###Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale using OpenCV.
As a last step, I normalized the image data such as the input distributions have mean zero, by applying the suggested (pixel - 128)/128. 

Traffic signs  before and after grayscaling & normalization

![alt text][image1]
![alt text][image2]

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolutional Layer 1 | 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12				    |
| Convolutional Layer 2	| 1x1 strides, valid padding, outputs 10x10x48  |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x48=1200				    |
| Flatten		        |        									    |
| Fully-connected Layer1| outputs 180        							|
| RELU					|												|
| DROPOUT				| dropout 60 %	keep probability				|
| Fully-connected Layer2| outputs 84        							|
| RELU					|												|
| DROPOUT				| dropout 60 %	keep probability				|
| Fully-connected Layer3| outputs 43        							|
|						|												|



The training epochs was originally set to 20 but I've noticed accuracy improvement if I increase the number of epochs, without posing a risk of overfitting the model (increased up to 50, although there is not much difference once 40 epochs are trained)
Haven't changed the learning rate (0.001). 

My final model results were:
* Validation Accuracy = 0.954
* Training Accuracy = 0.993
* Testing Accuracy = 0.935

LaNet architecture proved to be a good starting point as it provides accuracy up to 90%. However it didn't have the first 2 convolutional layers deep enough, 
while the fully connected layers size were not wide enough as a consequence, which proved difficult to accommodate the new more detailed dataset with far more variety. 
After doubling the outputs of the 1st convolutional layer I had to add dropout layers after each fully-connected layer, both with a 60% keep probability. 
Dropout layers helped reducing the amount of inputs for the following layers. 


###Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]

No image should be difficult to classify because they are all very similar to those within the training dataset.  

Here are the results of the prediction:

![alt text][image8]

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. Significantly better if compared to the original validation set, 
but of course with less probability to fail considering there are only 5 images within the dataset.




