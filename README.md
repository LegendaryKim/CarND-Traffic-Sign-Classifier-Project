# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images


[//]: # (Image References)

[image1]: ./writeupImages/index_samples.png "Samples"
[image2]: ./writeupImages/training_histogram.png "Histogram"
[image3]: ./writeupImages/original_preprocessing.png "Pre-processing"
[image4]: ./newImages_2/11.png "Traffic Sign 1"
[image5]: ./newImages_2/17.png "Traffic Sign 2"
[image6]: ./newImages_2/18.png "Traffic Sign 3"
[image7]: ./newImages_2/23.png "Traffic Sign 4"
[image8]: ./newImages_2/28.png "Traffic Sign 5"
[image9]: ./newImages_2/31.png "Traffic Sign 6"
[image10]: ./newImages_2/36.png "Traffic Sign 7"
[image11]: ./newImages_2/38.png "Traffic Sign 8"
[image12]: ./newImages_2/4.png "Traffic Sign 9"
[image13]: ./writeupimages_2/11_Top5.png
[image14]: ./writeupimages_2/18_Top5.png 
[image15]: ./writeupimages_2/18_Top5.png
[image16]: ./writeupimages_2/23_Top5.png
[image17]: ./writeupimages_2/28_Top5.png
[image18]: ./writeupimages_2/31_Top5.png
[image19]: ./writeupimages_2/36_Top5.png
[image20]: ./writeupimages_2/38_Top5.png
[image21]: ./writeupimages_2/4_Top5.png

---
### Data Set Summary & Exploration

#### Exploratory Data Analysis 

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?   34799
* The size of the validation set is ?   4410  
* The size of test set is ?   12630
* The shape of a traffic sign image is ?   32x32x3
* The number of unique classes/labels in the data set is ?   43


The examples of each labes in images are given as
![alt text][Image1]

The distribution of training set is 
![alt text][Image2]

### Design and Test a Model Architecture

#### Preprocessing techniques: histogram equalization, sharpening, and normalization.


As a first step in pre-processing, I applied histogram equalization to imporve the constrast of the images. And, I sharpend the image with the sharpening filter.

As a last step, I normalized the image data to distribute them with zero mean and unit variance. Here is an example of an original image and an pre-processed image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### Final Model Architecture

My final model consisted of the following layers:

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x3 RGB image                             | 
| Convolution 3x3       | 5x5 filter, 1x1 stride, same padding, outputs 32x32x6     |
| ELU                  |                                               |
| Max pooling           | 2x2 filter, 2x2 stride, valid padding, outputs 16x16x6                |
| Convolution 3x3       | 5x5 filter, 1x1 stride, same padding, outputs 16x16x16    |
| ELU                  |                                               |
| Max pooling           | 2x2 filter, 2x2 stride, valid pading, outputs 8x8x16              |
| Convolution 3x3       | 5x5 filter, 1x1 stride, same padding, outputs 8x8x20  |
| ELU                  |                                               |
| Max pooling           | 2x2 filter, 2x2 stride, valid padding, outputs 4x4x20                 |
| Flatten       | outputs 320                                           |
| Fully connected       | outputs 160                                           |
| Fully connected       | outputs 84                                        |
| Fully connected       | outputs 43                                        |
| Softmax               |                                            |


#### Training process

To train the model, the loss is calcuated with "softmax cross entorpy". The Adam optimizer is utilized.

The batch size is 50, the number of epochs is 20. The dropout, probability to keep units is 0.45. The mean and standard deviation of initilzied weights and biases is 0, 0.1 respectively. Lastly, the learning rate is 0.001.

#### Result

My final model results were:
* training set accuracy of ?   0.999
* validation set accuracy of ?   0.947 
* test set accuracy of ?   0.927

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

In my first tryout, I utilized the image data only with normalization and the identical structure with LeNet without dropout. The initial batch size was 100. The activation function was ReLU.

* What were some problems with the initial architecture?

The training set accuracy had stopped around 0.89 and the other accuracies are lower. It seems underfitting.


* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I added one more convolution layer with max pooling and fully connected layer in the network. Furthermore, the pre-processing was upgraded with histogram equalization and sharpening process. Also, the dropout layers are utilized just after fully-connected layers. My update increased the accuraices of all of my sets.

* Which parameters were tuned? How were they adjusted and why?
I decreased the size of batch to 50, and it improved the accuracies, too.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

The dropout layer helped to preventing overfitting. But, the my final results have still big difference between training and validation sets. If I have a chance, I want to augment the training and validation sample to reduce the overfitting effect.


If a well known architecture was chosen:
* What architecture was chosen?   Committee of CNNs by Team IDSIA
* Why did you believe it would be relevant to the traffic sign application? The multi-column deep neural network (MCDNN) is powerful with GPU environment to make final accurate prediction by averaging individual predictions of each DNN.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? The MCDNN with 25 DNN columns achieves 99.46%, it is better than the one of humans (98.84%). 


### Test a Model on New Images

#### Test with German traffic signs

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9] 
![alt text][image10] ![alt text][image11] ![alt text][image12] 

#### The mode's predictions

Here are the results of the prediction:

| Image                 |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection | Right-of-way at the next intersection | 
| No entry | No entry |
| General caution | General causion |
| Slippery road | Slippery road |
| Children crossing | Children crossing |
| Wild animals crossing | Wild animals crossing |
| Go straight or right | Go straight or right |
| Keep right | Keep right |
| Speed limit (70km/h)      | Speed limit (30km/h) |

The model was able to correctly guess 8 of the 9 traffic signs, which gives an accuracy of 88.89%. 

#### Predictions with softmax probability

![alt text][image13] ![alt text][image14] ![alt text][image15] 
![alt text][image16] ![alt text][image17] ![alt text][image18] 
![alt text][image19] ![alt text][image20] ![alt text][image21] 

