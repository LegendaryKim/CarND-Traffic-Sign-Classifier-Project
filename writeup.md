# **Traffic Sign Recognition** 
​
---
​
**Build a Traffic Sign Recognition Project**
​
The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report
​
​
[//]: # (Image References)
​
[image1]: ./writeupImages/index_samples.png "Samples"
[image2]: ./writeupImages/training_histogram.png "Histogram"
[image3]: ./writeupImages/original_preprocessing.png "Pre-processing"
[image4]: ./newImages/12.png "Traffic Sign 1"
[image5]: ./newImages/13.png "Traffic Sign 2"
[image6]: ./newImages/14.png "Traffic Sign 3"
[image7]: ./newImages/17.png "Traffic Sign 4"
[image8]: ./newImages/25.png "Traffic Sign 5"
[image9]: ./newImages/3.png "Traffic Sign 6"
[image10]: ./newImages/34.png "Traffic Sign 7"
[image11]: ./newImages/35.png "Traffic Sign 8"
[image12]: ./newImages/4.png "Traffic Sign 9"
[image13]: ./writeupImages/download(1).png
[image14]: ./writeupImages/download(2).png 
[image15]: ./writeupImages/download(3).png
[image16]: ./writeupImages/download(4).png
[image17]: ./writeupImages/download(5).png
[image18]: ./writeupImages/download(6).png
[image19]: ./writeupImages/download(7).png
[image20]: ./writeupImages/download(8).png
[image21]: ./writeupImages/download(9).png
​
---
### Data Set Summary & Exploration
​
#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.
​
I used the pandas library to calculate summary statistics of the traffic
signs data set:
​
* The size of training set is ?   34799
* The size of the validation set is ?   4410  
* The size of test set is ?   12630
* The shape of a traffic sign image is ?   32x32x3
* The number of unique classes/labels in the data set is ?   43
​
#### 2. Include an exploratory visualization of the dataset.
​
The examples of each labes in images are given as
![alt text][Image1]
​
The distribution of training set is 
![alt text][Image2]
​
### Design and Test a Model Architecture
​
#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)
​
As a first step in pre-processing, I applied histogram equalization to imporve the constrast of the images.
​
And, I sharpend the image with the sharpening filter.
​
As a last step, I normalized the image data to distribute them with zero mean and unit variance. 
​
Here is an example of an original image and an pre-processed image:
​
![alt text][image3]
​
The difference between the original data set and the augmented data set is the following ... 
​
​
#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.
​
My final model consisted of the following layers:
​
| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x3 RGB image                             | 
| Convolution 3x3       | 5x5 filter, 1x1 stride, same padding, outputs 32x32x6     |
| RELU                  |                                               |
| Max pooling           | 2x2 filter, 2x2 stride, valid padding, outputs 16x16x6                |
| Convolution 3x3       | 5x5 filter, 1x1 stride, same padding, outputs 16x16x16    |
| RELU                  |                                               |
| Max pooling           | 2x2 filter, 2x2 stride, valid pading, outputs 8x8x16              |
| Convolution 3x3       | 5x5 filter, 1x1 stride, same padding, outputs 8x8x20  |
| RELU                  |                                               |
| Max pooling           | 2x2 filter, 2x2 stride, valid padding, outputs 4x4x20                 |
| Flatten       | outputs 320                                           |
| Fully connected       | outputs 160                                           |
| Fully connected       | outputs 84                                        |
| Fully connected       | outputs 43                                        |
| Softmax               |                                            |
​
​
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
​
To train the model, the loss is calcuated with "softmax cross entorpy". The Adam optimizer is utilized.
​
The batch size is 50, the number of epochs is 20. The dropout, probability to keep units is 0.45. The mean and standard deviation of initilzied weights and biases is 0, 0.1 respectively. Lastly, the learning rate is 0.001.
​
#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.
​
My final model results were:
* training set accuracy of ?   0.987
* validation set accuracy of ?   0.931 
* test set accuracy of ?   0.909
​
If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
​
In my first tryout, I utilized the image data only with normalization and the identical structure with LeNet without dropout. The initial batch size was 100. The activation function was ReLU.
​
* What were some problems with the initial architecture?
​
The training set accuracy had stopped around 0.89 and the other accuracies are lower. It seems underfitting.
​
​
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
​
I added one more convolution layer with max pooling and fully connected layer in the network. Furthermore, the pre-processing was upgraded with histogram equalization and sharpening process. Also, the dropout layers are utilized just after fully-connected layers. My update increased the accuraices of all of my sets.
​
* Which parameters were tuned? How were they adjusted and why?
I decreased the size of batch to 50, and it improved the accuracies, too.
​
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
​
The dropout layer helped to preventing overfitting. But, the my final results have still big difference between training and validation sets. If I have a chance, I want to augment the training and validation sample to reduce the overfitting effect.
​
​
If a well known architecture was chosen:
* What architecture was chosen?   Committee of CNNs by Team IDSIA
* Why did you believe it would be relevant to the traffic sign application? The multi-column deep neural network (MCDNN) is powerful with GPU environment to make final accurate prediction by averaging individual predictions of each DNN.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? The MCDNN with 25 DNN columns achieves 99.46%, it is better than the one of humans (98.84%). 
 
​
### Test a Model on New Images
​
#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.
​
Here are five German traffic signs that I found on the web:
​
![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9] 
![alt text][image10] ![alt text][image11] ![alt text][image12] 
​
The first image might be difficult to classify because ...
​
#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).
​
Here are the results of the prediction:
​
| Image                 |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Priority road             | Priority road                                     | 
| Yield                 | Yield                                         |
| Stop                  | Stop                                          |
| No entry              | No entry                                  |
| Speed limit (60km/h)          | Speed limit (60km/h)                                  |
| Turn left ahead       | Turn left ahead                               |
| Ahead only        | Ahead only                                |
| Speed limit (70km/h)      | Speed limit (70km/h)                                  |
​
​
The model was able to correctly guess 9 of the 9 traffic signs, which gives an accuracy of 100%. 
​
#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)
​
![alt text][image13] ![alt text][image14] ![alt text][image15] 
![alt text][image16] ![alt text][image17] ![alt text][image18] 
![alt text][image19] ![alt text][image20] ![alt text][image21] 
​
### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
Omitted...
​