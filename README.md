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
[image9]: ./writeup_images/09_Sermanet.png "New training data distribution"
[image10]: ./writeup_images/10_FiveImages.png "Five new images"
[image11]: ./writeup_images/11_FiveImagesNorm.png "Five new images normalized"
[image12]: ./writeup_images/12_SpeedLimit.png "First image"
[image13]: ./writeup_images/13_SpeedLimitTop5.png "First image top 5"
[image14]: ./writeup_images/14_Stop.png "Second image"
[image15]: ./writeup_images/15_StopTop5.png "Second image top 5"
[image16]: ./writeup_images/16_RightofWay.png "Third image"
[image17]: ./writeup_images/17_RightofWayTop5.png "Third image top 5"
[image18]: ./writeup_images/18_Priority.png "Fourth image"
[image19]: ./writeup_images/19_PriorityTop5.png "Fourth image top 5"
[image20]: ./writeup_images/20_Bumpy.png "Fifth image"
[image21]: ./writeup_images/21_BumpyTop5.png "Fifth image top 5"
[image22]: ./writeup_images/22_ImageForVisualization.png "Image for visualization"
[image23]: ./writeup_images/23_ConvLayer1.png "Layer 1 output"
[image24]: ./writeup_images/24_ConvLayer2.png "Layer 2 output"

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

My final model implementation was a single stage model developed by Sermanet and LeCun and mentioned in this [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).

The figure from the paper below shows the multi-scale version of the model. 
![alt text][image9]
I've implemented a single scale version of the model, where the bottom connection from the 1st stage output directly to the classifier is missing. I've also added an additional fully connected layer with 100 hidden units and a dropout layer before the classifier.

My final model consisted of the following layers:

| Layer         					|     Description	        					| 
|:---------------------------------:|:---------------------------------------------:| 
| Input         					| 32x32x1 Grayscale image   					| 
| Convolution 5x5     				| 1x1 stride, valid padding, outputs 28x28x30 	|
| RELU								| Activation layer								|
| Max pooling	      				| 2x2 stride,  outputs 14x14x64 				|
| Convolution 5x5					| 1x1 stride, valid padding, outputs 10x10x64 	|
| RELU								| Activation layer								|
| Max pooling	      				| 2x2 stride,  outputs 5x5x64 					|
| Flattening	      				| input 5x5x64, 1600 outputs 					|
| Fully connected					| 100 outputs 									|
| RELU								| Activation layer								|
| Dropout							| 50% dropout during training					|
| Fully connected/Classifier		| 43 outputs 									|

Softmax is also applied to the final output logits as a part of the function `tf.nn.softmax_cross_entropy_with_logits()` used for training.

Below is the code for the function that sets up the model architecture.

``` python
def SermanetSS(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    conv1_f = 30
    conv2_f = 64
    fc1_nodes = 100
    classes = 43
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x30.
    conv1_w = tf.Variable(tf.truncated_normal([5, 5, 1, conv1_f], mean = mu, stddev = sigma), name = 'conv1_w')
    conv1_b = tf.Variable(tf.truncated_normal([conv1_f], mean = mu, stddev = sigma), name = 'conv1_b')
    conv1_stride = 1
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, conv1_stride, conv1_stride, 1], padding='VALID', name = 'conv1')
    conv1 = tf.nn.bias_add(conv1, conv1_b)
    # Activation.
    conv1 = tf.nn.relu(conv1, name = 'conv1_relu')
    # Pooling. Input = 28x28x30. Output = 14x14x30.
    k = 2
    conv1 = tf.nn.max_pool(conv1, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID', name = 'conv1_pool')
    # Layer 2: Convolutional. Output = 10x10x64.
    conv2_w = tf.Variable(tf.truncated_normal([5, 5, conv1_f, conv2_f], mean = mu, stddev = sigma), name = 'conv2_w')
    conv2_b = tf.Variable(tf.truncated_normal([conv2_f], mean = mu, stddev = sigma), name = 'conv2_b')
    conv2_stride = 1
    conv2 = tf.nn.conv2d(conv1, conv2_w, strides=[1, conv2_stride, conv2_stride, 1], padding='VALID', name = 'conv2')
    conv2 = tf.nn.bias_add(conv2, conv2_b)
    # Activation.
    conv2 = tf.nn.relu(conv2, name = 'conv2_relu')
    # Pooling. Input = 10x10x64. Output = 5x5x64.
    k = 2
    conv2 = tf.nn.max_pool(conv2, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID', name = 'conv2_pool')
    # Flatten. Input = 5x5x64. Output = 1600.
    flat = tf.contrib.layers.flatten(conv2)
    # Layer 3: Fully Connected. Input = 1600. Output = 100.
    fc1_w = tf.Variable(tf.truncated_normal([5*5*conv2_f, fc1_nodes], mean = mu, stddev = sigma), name = 'fc1_w')
    fc1_b = tf.Variable(tf.truncated_normal([fc1_nodes], mean = mu, stddev = sigma), name = 'fc1_b')
    fc1 = tf.add(tf.matmul(flat, fc1_w), fc1_b)
    # Activation.
    fc1 = tf.nn.relu(fc1, name = 'fc1_relu')
    # Dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    # Layer 4: Fully Connected. Input = 100. Output = 43.
    fc2_w = tf.Variable(tf.truncated_normal([fc1_nodes, classes], mean = mu, stddev = sigma), name = 'fc2_w')
    fc2_b = tf.Variable(tf.truncated_normal([classes], mean = mu, stddev = sigma), name = 'fc2_b')
    fc2 = tf.add(tf.matmul(fc1, fc2_w), fc2_b)
    logits = fc2
    return logits
```

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer. Here's the code that sets up the optimizer and loss function.

``` python
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
dropout = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 43)
logits = SermanetSS(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
```

The code to execute the training:

``` python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train_n)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train_n, y_train = shuffle(X_train_n, y_train)
        # Train    
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_n[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, dropout: keep_prob})
```

The hyperparameters I've selected are:

``` python
    EPOCHS = 60
    BATCH_SIZE = 100
    rate = 0.0005
    keep_prob = 0.5
```

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

First, the architecture I've used was the same LeNet model from the digit classification example. The only change that I made was to extend the number of classes from 10 to 43. Using this model for traffic sign recognition makes sense as the task of classifing handwritten digits is very similar with the exception of having more classes to choose from. 
The sizes of inputs and outputs of layers of this model were the following:

32x32x1 -> 28x28x6 -> 14x14x6 -> 10x10x16 -> 5x5x16 -> 400 -> 120 -> 84 -> 43.

It was able to achieve 94.5% validation accuracy. This model was trained on the original training data set, so the class distribution of data was uneven.

Next, I've decided to implement a model from the [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) by Sermanet and LeCun. They were able to achieve accuracy of above 98%, so I was hoping to get close to that number. I wanted to start with the simpler architecture first, which is the single scale model. It's pretty much the same LeNet model architecture, with additional features (30 and 64) in each convolution layer. Again, it makes sense to use the same or similar architecture as it's proven to be effective on a similar task. Extending the number of features in each layer should help because of increased number of classes.
The sizes of inputs and outputs of layers of this model were the following:

32x32x1 -> 28x28x30 -> 14x14x64 -> 10x10x64 -> 5x5x64 -> 1600 -> 100 -> 100 -> 43

A change from orignial LeNet to this model resulted in improved validation accuracy of 96.5%.

The next thing I wanted to try was to improve the distribution of data across the classes, so I've generated additional images to make the minimum number of samples per class at first 500 and then 1500. The validation accuracy changed to 97.5% and 96.4% respectively. Model architecture remained the same as above. I assumed that the drop in accuracy happened because of variation in model parameter initialization and not an inherent issue with having more data to train on. 

The last change to the model that I made was adding a dropout layer and removing one of the fully connected layers. Having a dropout layer helps improve model's accuracy by having it essentialy try to classify the image using only part of the model's architecture.
This resulted in the final model architecture as mentioned in the previous section.
The sizes of inputs and outputs of layers of this model were the following:

32x32x1 -> 28x28x30 -> 14x14x64 -> 10x10x64 -> 5x5x64 -> 1600 -> 100 -> 43. 

This model resulted in validation accuracy of 97.5%.

At this point I was happy with the results as they were close to the ones reported in the aforementioned paper and decided to test the model on the training set and a test set.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 97.5%
* test set accuracy of 96.7%

This proved that model was well trained and the architecture of the model was selected successfully for the task of traffic sign classification. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image10]

They are not particularly difficult images to classify, but they were resized from the original image sizes down to a 32x32 image, so this may cause some issues for the classifier as the sign proportions may be off.

I manually provided a class label for each image, converted them to grayscale and normalized to match the expected neural network input.

![alt text][image11]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image					        |     Prediction	        					| 
|:-----------------------------:|:---------------------------------------------:| 
| Speed limit 70 km/h      		| Speed limit 70 km/h							| 
| Stop      					| Stop 											|
| Right-of-way					| Right-of-way									|
| Priority road   				| Priority road					 				|
| Bumpy road					| Bumpy road      								|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 96.7%. These images were not particularly difficult and the proportions were only slightly distorted, so it makes sense that the model was able to correctly predict all of them.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Here's my code for top 5 softmax probabilities for each image:

``` python
with tf.Session() as sess:
    saver.restore(sess, "model.ckpt")
    for idx in range(len(images_norm)):
        (top_prob, top_classes) = sess.run(top_five, feed_dict={x: [images_norm[idx]], y: images_y[idx], dropout: 1})
        np.set_printoptions(precision=2, suppress=True)
        plt.figure((idx*2), figsize=(2,2))
        plt.imshow(images_rgb[idx].squeeze())
        plt.figure((idx+1)*2-1, figsize=(10,10))
        for cls in range(len(top_classes[0])):
            sub = plt.subplot(1,5,cls+1)
            class_indices = np.where(y_test == top_classes[0][cls])
            class_sample = class_indices[0][0]
            plt.imshow(X_test[class_sample])
            sub.set_title('Probability: {:.1f}%'.format(top_prob[0][cls]*100))
```

For each image, I've decided to plot sample images of the top 5 classes with the probability percentage value shown on top of them.

For the first image (70 km/h speed limit sign), the model is absolutely sure it's a 70 km/h speed limit sign (probability of 100%).

![alt text][image12]
![alt text][image13]

The next best candidate is a 20 km/h speed limit sign. It makes sense as a 2 looks similar to a 7, but the probability of that is very small.

For the second image (stop sign), the model is absolutely sure it's a stop sign (probability of 100%).

![alt text][image14]
![alt text][image15]

The other options do not make much sense to me, but their probabilites are very very low.

For the third image (right-of-way sign), the model is again absolutely sure it's a right-of-way sign (probability of 100%).

![alt text][image16]
![alt text][image17]

The other 4 options do have some features in common with the actual sign, such as triangular shape of the sign and the shape in the middle of the sign that looks similar. Their probabilites are still very very low, so the model is very certain about the prediction.

For the fourth image (priority road sign), the model is absolutely sure it's a priority road sign (probability of 100%).

![alt text][image18]
![alt text][image19]

The other options do not make much sense to me, but their probabilites are very very low.

For the final image (bumpy road sign), the model is absolutely sure it's a bumpy road sign (probability of 100%).

![alt text][image20]
![alt text][image21]

Options 2,3, and 4 do have some features in common with the actual sign, such as triangular shape of the sign and the shape in the middle of the sign that looks similar. The last of the top 5 looks quite different. In any case, their probabilites are very low, so the model is very certain about the correct prediction.

Overall, the model is very confident about each of the image's predicitons and it has the right to be as it predicted the class of every one of the images correctly!

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
To get an idea of what features neural network is looking for in an image, I've run the following code that plots all the feature maps for both convolutional layers:

``` python
def outputFeatureMap(sess, image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x: image_input, dropout: 1})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(10,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")

with tf.Session() as sess:
    saver.restore(sess, "model.ckpt")
    
    #print([tensor.name for tensor in tf.get_default_graph().as_graph_def().node])
    img_num = random.randrange(n_test)
    plt.figure(1)
    plt.imshow(X_test_n[img_num].squeeze(), cmap='gray')
    plt.title(y_test[img_num])
    
    tens1 = sess.graph.get_tensor_by_name('conv1_relu:0')
    tens2 = sess.graph.get_tensor_by_name('conv2_relu:0')
    
    outputFeatureMap(sess, [X_test_n[img_num]], tf_activation = tens1, plt_num = 2)
    outputFeatureMap(sess, [X_test_n[img_num]], tf_activation = tens2, plt_num = 3)
```

The image that was randomly selected is an end of 80 km/h speed limit sign.

![alt text][image22]

The output of the first convolutional layer is shown below:

![alt text][image23]

From the output, some important features can be seen. For example, feature map 22 shows that the network is looking for a circular shape of the sign and the number 80. Number 80 can also be seen in feature map 1. The the diagonal line with a positive slope crossing the sign shows up clearly in feature maps 8 and 26. A lot of the feature maps highlight that the sign is circular in shape.

The output of the second convolutional layer is shown below:
![alt text][image24]

It's much harder to understand what is being shown here, but the diagonal lines with a positive slope and circular shapes (that could potentially be combined to make number 80) are present here as well.

Overall, it's not always perfectly clear how the neural network decides on what sign it's seeing, but it does give a bit of an insight into it.