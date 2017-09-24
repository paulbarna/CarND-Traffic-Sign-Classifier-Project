#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

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

[image0]: ./BarChart.png

###Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale using OpenCV.
As a last step, I normalized the image data such as the input distributions have mean zero, by applying the suggested (pixel - 128)/128. 

Traffic signs  before and after grayscaling & normalization

[image1]: ./Original.png
[image2]: ./PreProcess.png


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



The training epochs was originally set to 20 but I've noticed accuracy improvement if I increase the number of epochs (40), without posing a risk of over-training the model.
Haven't changed the learning rate (0.001).

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

LaNet architecture proved to be a good starting point as it provides accuracy up to 90%. However it didn't have the first 2 convolutional layers deep enough, 
while the fully connected layers size were not wide enough as a consequence, which proved difficult to accommodate the new more detailed dataset with far more variety. 
After doubling the outputs of the 1st convolutional layer I had to add dropout layers after each fully-connected layer, both with a 60% keep probability. 
Dropout layers helped reducing the amount of inputs for the following layers. 


###Test a Model on New Images

Here are five German traffic signs that I found on the web:

[image3]: ./WebSigns/1.jpg "Speed limit (30km/h)"
[image4]: ./WebSigns/11.jpg "Right-of-way at the next intersection"
[image5]: ./WebSigns/12.jpg "Priority road"
[image6]: ./WebSigns/17.png "No entry"
[image7]: ./WebSigns/35.png "Ahead only"

No image should be difficult to classify because they are all very similar to those within the training dataset.  

Here are the results of the prediction:

WebSigns\1.jpg (Speed limit)

Top five softmax probabilities:
   1 - Speed limit (30km/h) : 100.0 %
   2 - Speed limit (50km/h) : 0.0 %
   4 - Speed limit (70km/h) : 0.0 %
   0 - Speed limit (20km/h) : 0.0 %
   21 - Double curve : 0.0 %

WebSigns\11.jpg (Right-of-way at the next intersection)

Top five softmax probabilities:
   11 - Right-of-way at the next intersection : 100.0 %
   30 - Beware of ice/snow : 0.0 %
   21 - Double curve : 0.0 %
   27 - Pedestrians : 0.0 %
   40 - Roundabout mandatory : 0.0 %

WebSigns\12.jpg (Priority road)

Top five softmax probabilities:
   12 - Priority road : 99.8 %
   13 - Yield : 0.2 %
   15 - No vehicles : 0.0 %
   8 - Speed limit (120km/h) : 0.0 %
   38 - Keep right : 0.0 %

WebSigns\17.jpg (No entry)

Top five softmax probabilities:
   17 - No entry : 100.0 %
   14 - Stop : 0.0 %
   0 - Speed limit (20km/h) : 0.0 %
   34 - Turn left ahead : 0.0 %
   40 - Roundabout mandatory : 0.0 %

WebSigns\35.jpg (Ahead only)

Top five softmax probabilities:
   35 - Ahead only : 100.0 %
   33 - Turn right ahead : 0.0 %
   13 - Yield : 0.0 %
   36 - Go straight or right : 0.0 %
   12 - Priority road : 0.0 %

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. Significantly better if compared to the original validation set, 
but of course with less probability to fail considering there are only 5 images within the dataset.




