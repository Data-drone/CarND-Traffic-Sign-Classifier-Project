# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[train_image]: ./report_images/train.png "Train data"
[test_image]: ./report_images/test.png "Test data"
[validation_image]: ./report_images/validation.png "Validation data"
[example_image]: ./report_images/images.png "Image data"
[resample_example_image]: ./report_images/resample_processing.png "Resample data"

[new_image1]: ./new_images/Priority-768x765.jpg "New image 1"
[new_image2]: ./new_images/32842016-german-road-sign-give-way.jpg "New image 2"
[new_image3]: ./new_images/3677862-speed-limit-sign-in-germany.jpg "New image 3"
[new_image4]: ./new_images/9.jpg "New image 4"
[new_image5]: ./new_images/100_1607.jpg "New image 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

The number of items in each class is uneven.

For the training data it is:

![training image][train_image]

For the validation data it is:

![validation image][validation_image]


For the test data it is:

![test image][test_image]


The full set of example images for each class looks like:

![example image][example_image]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Looking at the dataset, some classes are not represented much. In order to fit these samples better, I resampled the dataset in order to have 500 images of each class as a minimum. But to include these resampled images direct would likely lead to overfitting to I added noise and random rotations to the resampled images.

Here is an example of the resample pre processing and results for each class:

![resample processing][resample_example_image]

As a last step, I normalized the image data so that the number values seen by the neural network is more regular and the weights won't overfit on certain images due to their numerical values.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is a standard LeNet with dropout on the Fully Connected Layers 
The structure is the following:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Fully connected		| outputs 120        							|
| RELU  				|           									|
| Dropout				| keep prob 35%									|
| Fully connected		| outputs 84			    					|
| RELU  				|           									|
| Dropout				| keep prob 35%									|
| Fully connected		| outputs 43			    					|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The model was trained using an Adam Optimiser with a learning rate of 0.009 for 5450 epochs and a batch size of 256. At first I trained with a high learning rate to make sure that the network was training smoothly and that the model was learning. In order to achieve the final 0.93 validation accuracy, however, I used a small learning rate with more epochs this reduces the chance that the model overfits on particular data samples.

The dropout settings for training were hand tuned until the training accuracy and validation accuracy moved together in sync as opposed to the training accuracy rapidly rising and the validation accuracy staying still as was the case when training without drop out. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.904
* validation set accuracy of 0.938 
* test set accuracy of 0.925

The standard LeNet architecture proved sufficiant to achieve the results required. Prior to the addition of dropout it did overfit however. The dataset as provided also proved insufficient hence the resampling procedure with noise and random rotations added.

The process I undertook was to watch the validation and train accuracies for each model run. Should the train accuracy be much higher than the validation accuracy, I reduced the keep prob on the dropout layers in order to reduce overfitting.

For each model run, I experimented with different data augmentation settings in order to extract the most performance out of the LeNet architecture.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![sign 1][new_image1] ![sign 2][new_image2] ![sign 3][new_image3] 
![sign 4][new_image4] ![sign 5][new_image5]

The third sign proved difficult to identify due to it's similarity to other speed limit signs. The distinguishing factor is the '5' but particularly when reduced to 32x32 resultion it looks similar to '3' 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority Road     	| Priority Road   								| 
| Yield     			| Yield 										|
| 50 km/h				| 30 km/h										|
| Roundabout mandatory	| Roundabout mandatory					 		|
| Right of way at the next intersection	| Right of way at the next intersection      |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is not as good as the test set accuracy of 92.5% but the new traffic sign test is a small dataset with more data it will likely exceed 80% accuracy

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located under the heading Predict New Images towards the end of the Ipython notebook.

For the first image, the model is relatively sure that this is a Priority Road sign (probability of 1.0), and the image does contain a Priority Road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Priority Road   								| 
| 0.0     				| No passing for vehicles over 3.5 metric tons	|
| 0.0					| Roundabout mandatory							|
| 0.0	      			| No entry					 				    |
| 0.0				    | Traffic signals      							|

For the second image, the model is relatively sure that this is a Yield Road sign (probability of 1.0), and the image does contain a Yield Road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Yield            								| 
| 0.0     				| Ahead Only                                	|
| 0.0					| Speed limit (20km/h)							|
| 0.0	      			| Speed limit (30km/h)					 	    |
| 0.0				    | Speed limit (50km/h)      					|

For the third image, the model is sure that this is a Speed limit (30km/h) sign (probability of 0.9), but the image does not contain a Speed limit (30km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.96         			| Speed limit (30km/h)            				| 
| 0.03     				| Speed limit (70km/h)                          |
| 0.006					| Speed limit (50km/h)							|
| 0.002	      			| Speed limit (20km/h)					 	    |
| 0.0				    | No entry                    					|

For the fourth image, the model is sure that this is a Roundabout mandatory sign (probability of 0.9), and the image does contain a Roundabout mandatory sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.98         			| Roundabout mandatory            				| 
| 0.009     			| Go straight or left                          |
| 0.003					| Turn left ahead							|
| 0.0	      			| Go straight or right					 	    |
| 0.0				    | Keep left                    					|

For the fifth image, the model is sure that this is a Right-of-way at the next intersection sign (probability of 0.9), and the image does contain a Right-of-way at the next intersection sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| Right-of-way at the next intersection         | 
| 0.0     				| Beware of ice/snow                            |
| 0.0					| Traffic signals							    |
| 0.0	      			| Double curve					 	            |
| 0.0				    | Pedestrians                    				|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


