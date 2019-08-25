# **Traffic Sign Recognition** 
# by Dmitriy Semenov

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Download the data set (if it doesn't exist already)
* Load the data set
* Explore, summarize and visualize the data set
* Pre-process the data set
* Generate additional data to normalize class distribution
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Visualize the Neural Network's convolutional layers


[//]: # (Image References)

[image1]: ./writeup_images/01_visualize.png "Visualization"
[image2]: ./writeup_images/02_TrainDistr.png "Training data distribution"
[image3]: ./writeup_images/03_ValidDistr.png "Validation data distribution"
[image4]: ./writeup_images/04_TestDistr.png "Test data distribution"
[image5]: ./writeup_images/05_Normalized.png "Grayscaled and normalized"
[image6]: ./writeup_images/06_Modified.png "Modified image"
[image7]: ./writeup_images/07_NewData.png "New training data distribution"
[image8]: ./writeup_images/08_SampleNewImages.png "New training data distribution"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! Here is a link to my [project code](Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration
#### Data download and loading.
Before anything can be done to the image data set, it needs to be downloaded and loaded in memory.
I used zipfile, urllib3, and os libraries to do that. Code also checks to see if the file already exists before downloading it.

``` python
import zipfile
import urllib3
import os

data_url = 'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip'
zipFilename = 'image_data/traffic-signs-data.zip'

# check for extraction directories existence
if not os.path.isdir('image_data'):
    os.makedirs('image_data')
    
if not os.path.isfile(zipFilename):
    http = urllib3.PoolManager()
    
    # download data
    print("Downloading ",data_url)
    response = http.request('GET', data_url)
    zippedData = response.data
    
    # save data to disk
    print("Saving to ",zipFilename)
    output = open(zipFilename,'wb')
    output.write(zippedData)
    output.close()
    
# extract the data
zfobj = zipfile.ZipFile(zipFilename)
for file in zfobj.namelist():
    outputFilename = "image_data/" + file
    if not os.path.isfile(outputFilename):
        uncompressed = zfobj.read(file)
        # save uncompressed data to disk
        print("Saving extracted file to ",outputFilename)
        output = open(outputFilename,'wb')
        output.write(uncompressed)
        output.close()

print("Done with download and unzip")
```

After data is unzipped, it's extracted using pickle library, and stored in X and y variables for training, validation, and test data sets and labels. The pickled data is a dictionary with 4 key/value pairs:

* 'features' is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
* 'labels' is a 1D array containing the label/class id of the traffic sign. The file signnames.csv contains id -> name mappings for each id.
* 'sizes' is a list containing tuples, (width, height) representing the original width and height the image.
* 'coords' is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES

``` python
# Load pickled data
import pickle

training_file = 'image_data/train.p'
validation_file= 'image_data/valid.p'
testing_file = 'image_data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

print("Done with data set loading")
```
#### 1. Provide a basic summary of the data set.

Next, to get an overview of the data, I used the python and numpy to calculate summary statistics of the traffic
signs data set:

``` python
import numpy as np
### Use python, pandas or numpy methods rather than hard coding the results

# Number of training examples
n_train = X_train.shape[0]

# Number of validation examples
n_validation = X_valid.shape[0]

# Number of testing examples.
n_test = X_test.shape[0]

# What's the shape of an traffic sign image?
image_shape = (X_test[0].shape)

# How many unique classes/labels there are in the dataset.
n_classes = np.max(y_train)-np.min(y_train)+1

```
Here are the statistics:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

I used matplotlib library to visualize some sample images in the dataset and also to show the distribution of data between classes in each data set.

``` python
### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import random
# Visualizations will be shown in the notebook.
%matplotlib inline

# Plot some images of different classes
img_count = 5
plt.figure(1, figsize=(15,15))
for i in range(img_count):
    sub = plt.subplot(1,img_count,i+1)
    img_num = random.randrange(n_test)
    plt.imshow(X_test[img_num])
    sub.set_title(y_test[img_num])

# Plot distribution of each class in each set
fig2 = plt.figure(2)
fig2.suptitle('Training data distribution')
plt.hist(y_train, bins=n_classes)

fig3 = plt.figure(3)
fig3.suptitle('Validation data distribution')
plt.hist(y_valid, bins=n_classes)

fig4 = plt.figure(4)
fig4.suptitle('Test data distribution')
plt.hist(y_test, bins=n_classes)
```

Here are some of the example images from the data.

![alt text][image1]

Another important aspect of data is how it's distributed between classes. Images below show the distribution.

![alt text][image2]
![alt text][image3]
![alt text][image4]

It can be seen that some of the classes contain a lot more data than others. This may be a problem for neural network while training as it may force the weights and biases of the network to be scewed towards recognizing images of the more represented classes. We will address this issue in one of the future steps by generating additional image data for classes that are not well represented.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

To pre-process image data I've chosen two methods: converting rgb images to grayscale and normalizing data, so that the data has mean zero and equal variance. Here are the two functions to do that. I've used OpenCV's `cvtColor()` function to convert to grayscale.

``` python
def grayscale(image_data):
    gray = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in image_data]
    gray = np.array(gray)
    gray = gray[:,:,:,np.newaxis]
    return gray

def normalize(image_data):
    norm = (image_data-128.0)/128
    return norm
```

Images below are some of the samples showing the effect of grayscaling and normalization.
![alt text][image5]

Additionally, I've created a `modify()` function, which is used to generate new images out of existing ones. This function randomly shifts, rotates, and scales an image by a small amount.

``` python
def modify(img):
    rows,cols = img.shape
    
    # Shift
    x_shift = random.randrange(-2,2,1)
    y_shift = random.randrange(-2,2,1)
    M_shift = np.float32([[1,0,x_shift],[0,1,y_shift]])
    shifted_img = cv2.warpAffine(img,M_shift,(cols,rows))
    
    # Rotation & Scaling
    angle_rot = random.randrange(-15,15,1)
    scale = random.uniform(0.9,1.1)
    M_rot = cv2.getRotationMatrix2D((cols/2,rows/2),angle_rot,scale)
    rotated_img = cv2.warpAffine(shifted_img,M_rot,(cols,rows))
    outimg = rotated_img[:,:,np.newaxis]
    return outimg
```
Here is an example of the output of this function applied to a normalized image.
![alt text][image6]

Values for the amount of shift, rotation, and scaling were based on the numbers mentioned in [Pierre Sermanet and Yann LeCun's paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).

As mentioned earlier, I've decided to generate additional training images to make class distribution more equal.
I've decided to make the minimum number of samples per class to be 1500.

``` python
filename_train_n = 'image_data/train_n.p'
if os.path.isfile(filename_train_n):
    with open(filename_train_n, mode='rb') as f:
        train_n = pickle.load(f)
    X_train_n, y_train = train_n['features'], train_n['labels']
    print('Finished loading normalized data')
else:
    targetimgcount = 1500
    print('Started generating additional data for image classes.')
    for class_label in range(n_classes):
        class_indices = np.where(y_train == class_label)
        n_samples = len(class_indices[0])
        if n_samples < targetimgcount:
            print('Starting sample count for class label ',class_label, ': ', n_samples)
            for i in range(targetimgcount - n_samples):
                idx = class_indices[0][i % n_samples]
                generatedimg = modify(X_train_n[idx].squeeze())
                X_train_n = np.append(X_train_n, [generatedimg], axis = 0)
                y_train = np.append(y_train, class_label)
            print('Finished generating images for class label ',class_label)
    print('Finished generating additional data for image classes.')
```

This brought the total number of training images to 67380.
Here's how the distributions of training data compare after additional images were generated.

| Orignial Distribution         		|     New Distribution	        					| 
|:-------------------------------------:|:-------------------------------------------------:| 
![alt text][image2]                     |![alt text][image7]


I decided to visualize some of the generated images to make sure that the newly generated images were valid.
``` python
gen_img_count = X_train_n.shape[0] - X_train.shape[0]
img_count = 5
plt.figure(1, figsize=(15,15))
for i in range(img_count):
    sub = plt.subplot(1,img_count,i+1)
    img_num = X_train.shape[0] + random.randrange(gen_img_count)
    plt.imshow(X_train_n[img_num].squeeze(),cmap='gray')
    sub.set_title('idx: ' + str(img_num) + ' class: ' + str(y_train[img_num]))
```

Here are some of the examples. Their index shows that these are newly created images, since they all have index above 34799.
![alt text][image8]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
