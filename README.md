# NDBuildingRecognition

# Part 1 - Conceptual Design
# Authors: Anna Muller and Patrick Hsiao

First, our main goal of this project is being able to identify what building an image contains. In order to do this, we have to figure out what the building is. At this point, we don’t have a dataset that contains the buildings we are trying to identify, so that’s our first step. We need to create a dataset of different buildings we need to highlight. Something we have to think about with the buildings are the picture settings: lighting, brightness, scale, weather, background clutter, season, contrast, interior or exterior, etc.. It will be helpful to capture images at different angles and different times of the day, and with different phone cameras. There’s a lot of moving parts and we’ll have to decide what we want to account for and what we don’t want to account for. 

To get this dataset, we will actually go to each building we are trying to identify and take multiple pictures with differing picture settings. However, we have to really think about how many different settings we include because the more we include, the harder it will be for the model to learn which building is which. 
With this dataset in mind, our goal is for the ratio to be 80% for training, 10% for validation and 10% for testing. We will then randomly assign different building pictures to either the training, validation or test groups using a python script. Something else we have to keep in mind is whether a building will be excluded from the training, validation or testing phases, we probably would want all buildings to be included in the training set so that we could have a more accurate model.
After our data is organized, our next step will be to start writing the code for the building recognition algorithm. First we will need to load the data into the program, then decide what features to calculate. We could use color histograms to capture the distribution of colors in the image. This can help in distinguishing buildings based on color patterns. We could also use texture analysis (e.g., Gabor filters or texture energy measure) to quantify the texture characteristics of building facades. We could also do edge detection algorithms to identify shape edges and boundaries of buildings. We could detect corners and keypoints in the image using algorithms like Harris corner detection or SIFT (Scale-Invariant Feature Transform). This can help in matching distinctive building features. We could analyze the shapes within the image to identify geometric patterns and architectural styles. This might involve techniques that analyze contour or do shape matching. 

With all the features extracted, we can proceed to develop the building recognition model. This model will take the extracted features as input and learn to associate them with specific buildings. In order to do this, we have to decide what kind of architecture we will use. We could use a convolutional neural network to do this. Ultimately, our biggest task is to determine what features we want to really focus on and how we capitalize on this. We may have to experiment with a bunch of different features to really figure out what matters the most in identifying buildings. Once we do this, we can start to code the algorithm and use a CNN architecture to be able to accomplish this task. 

Division of Labor: 
Overview 1/2 Patrick , 1/2 Anna
Dataset Discussion: 1/2 Patrick, 1/2 Anna
Algorithms Discussion: 1/2 Patrick, 1/2 Anna

# Part 2 - Data Set

We collected images of 5 buildings on campus: Lyons Hall, South Dining Hall, Hesburgh Library, the Main Building, and Knott Hall. We collected 100 images of each building, at different angles, captured at different times of the day (morning and evening).
We plan to split the images three ways for the training set, the validation set, and the training set, evenly dispering both evening and night images. The training set will exhibit diversity so that we can train the model to recognize the buildigns from many different angles and times of day, and the validation set will verify that this is indeed possible. 
Some of the images show the buildings in their entirety, while some of the images are cut off. The resolution of the samples were all taken on iPhone cameras, and the conditions were relatively clear out (for instance, we avoided taking pictures on days where it was raining very heavily)

Division of Labor:
1/2 of images were taken by Patrick, 1/2 of images were taken by Anna

# Part 3 - Preprocessing and Feature Extraction

**List of methods applied for pre-processing and feature extraction:**
- convert colored images to grayscale images
- compute keypoints and descriptors
- match descriptors across images

**SIFT Justification**

The SIFT algorithm, or Scale-Invariant Feature Transform, is an algorithm that extracts distinctive features from images, despite any scale, rotation, and lighting chagnes in the images. The SIFT algorithm identifies key points within an image and computes a descriptor for each point. The descriptor can then match and recognize objects in different images.

We believed that SIFT would be the best algorithm to use for our building recognition program because it can work well despite changes in scale and rotation. Many of our images of Notre Dame buildings are taken from different angles, and thus appear to have different sizes and rotations. SIFT is able to detect and describe key features of the building despite these setbacks.

In addition, SIFT is very good at distinctive feature extraction. Many of the buildings that we photographed have very distinct architectural features (such as the Dome, the arch in Lyons Hall, and the windows in South dining hall). We wanted to leverage SIFT's ability to extract these distinct features so that we can most effectively identify these buildings.

SIFT also performs very well despite changes in lighting. Half of our data set was taken in the morning, and half was taken in the afternoon. Therefore, the lighing in the data set varies greatly. However, SIFT focuses on the structure of the image rather than the value of the pixels, so the performance of the algorithm is not greatly affected by this. 

Also, many of the building images in our data set do not capture the building entirely. Some of our images are blocked by trees and pedestrians, and in some images the photographer simply did not include the entire building in the shot. However, SIFT is very good at handling partial views, as it can match the key features that are visible even when other key features are hidden.

SIFT is also very good at creating a reference database of key features. When recognizing an unknown building, SIFT can easily generate distinctive descriptors, and match those descriptors to the most similar descriptors in the database. SIFT is also very scalable. Although the dataset is currently relatively small, if we ever wanted to expand it in the future, SIFT would make it easy to do so.

SIFT is also used in many fields that span beyond just building recognition. It can be used in more advanced cases, like facial recogntiion, scene recognition, and object tracking - its versatility is very impressive. 

Importantly, open source implementations can easily be accessed. In our case, we were able to access it using the open CV library. The open source nature of this algorithm encourages collaborative improvements, ensuring that the algorithm adapts to evolving requirements.

**Method Illustrations**

Original image:

![image](https://github.com/hsiaopat/NDBuildingRecognition/assets/97554902/50419704-5cd9-472b-b7ba-5cc0cae6704b)

Image after identifying key points:

![image](https://github.com/hsiaopat/NDBuildingRecognition/assets/97554902/a63bbcee-5153-46d8-b762-fd9041106543)

Original images:

![image](https://github.com/hsiaopat/NDBuildingRecognition/assets/97554902/1f0baf12-22be-4b1a-9bac-6c0f16c537b3) ![image](https://github.com/hsiaopat/NDBuildingRecognition/assets/97554902/d6f712cd-fbe5-44f9-8d31-864a9b544a5e)

Images with key point matching: 

![image](https://github.com/hsiaopat/NDBuildingRecognition/assets/97554902/07db31ac-5a99-479f-8c4e-85b085a95057)

Now that we have the pre-processing dealt with, we can now employ different techniques utilizing this data that we have to actually implement the classification. We have discussed various models and strategies to use with our pre-processed data and some of our research is highlighted below:

Deep Learning:
One of our approaches involves leveraging deep learning models for image classification. Deep neural networks, especially Convolutional Neural Networks (CNNs), have demonstrated remarkable performance in image-related tasks. By feeding the SIFT features or the pre-processed images directly into the neural network, we can harness the power of hierarchical feature learning. Fine-tuning a pre-trained model on our specific dataset is an option, which can be particularly beneficial when dealing with limited data.

Bag of Visual Words (BoVW):
We could utilize the Bag of Visual Words model. This involves converting the SIFT key points into a visual vocabulary by clustering them into visual words. Each image is then represented as a histogram of these visual words. This method, rooted in traditional computer vision techniques, is robust and can be effective for classification tasks. It provides a structured way to capture the spatial distribution of key features in the images.

Support Vector Machines (SVMs):
SVMs offer a classical machine learning approach for image classification. The SIFT features, transformed into feature vectors, can be used to train an SVM classifier. SVMs are particularly suitable for scenarios where the number of features is not excessively large compared to the number of samples. With proper hyperparameter tuning, SVMs can effectively delineate decision boundaries in feature space, aiding in the classification of different types of building images.

Ensemble Approach:
Recognizing the potential strengths of each individual technique, we will explore the possibility of creating an ensemble model. This involves combining the predictions from different models, such as deep learning, BoVW, and SVMs. Ensemble methods often lead to improved overall performance by mitigating the weaknesses of individual models and leveraging their collective strengths. By aggregating these diverse approaches, we aim to enhance the robustness and generalization capability of our final classification model.

This approach allows for a comprehensive exploration of various techniques, ensuring a thorough understanding of their individual strengths and weaknesses. The subsequent aggregation of results will enable us to determine the most effective model for our specific use case, striking a balance between accuracy, interpretability, and computational efficiency.

**Division of Labor**

1/2 Pre-processing work done by Patrick, 1/2 Pre-processing work done by Anna

1/2 of write-up done by Patrick, 1/2 write-up done by Anna

# Part 4 - Classification
Note: Used late token extension

**Choice of Classifier**

To classify our data, we used Kullback-Leibler (KL) divergence. We have already created a labeled dataset in which each instance is a different building on campus. We then computed the probability distribution of features relevant to each building in the training dataset. Then, we used KL divergence to measure the dissimilarity between the feature distributions of different buildings on campus. We were then able to validate the perforamnce of KL divergence-based classifier using the validation set.

KL Divergence was a good choice for this specific project because it is very well-suited for capturing dissimilarities in feature distributions, which helps differentiate each building by any unique features that the buildings may have. 

In addition, by treating the features as probabilities, we are able to measure the divergence between the expected and observed feature distributions for different buildings to get an accurate read on how confident our model is about its selections.

In addition, KL divergence can be incorprated into many different classification methods, giving us flexibility to change around our classification method in case we were not able to attain a satisfactory classiication of our buildings.

**Classification Accuracy**

Training Subset Classification Accuracy: 15%

Validation Subset Classification Accuracy: 25%

**Commentary**

Our accuracy was lower than we anticipated. Reasons for this include images in the training set that do not have enough connectivity.
For example, the below images are both images of Hesburgh Library. However, they look as though they could be completely different buildings. For the final sprint, we will consider refining the images that we use and possibly take new ones if necessary in order to better train the model.

![IMG_8408 Small](https://github.com/hsiaopat/NDBuildingRecognition/assets/97554902/63eaf917-f493-4dc7-a121-cc7dfcb97745) ![IMG_8402 Small](https://github.com/hsiaopat/NDBuildingRecognition/assets/97554902/bb5aff9c-a009-47f4-95e9-3f9542dc47b2)

In addition, another factor that contributes this is similar architecture between different buildings. Buildings at the University of Notre Dame follow a similar theme, and thus there are higher chances of certain architectural features getting mistaken for other buildings.

One thing to note is that the validation subset classification accuracy is higher than the training subset classification. Typically, the training subset accuracy is expected to be higher. The reason for our results might be because our subset is very small - this could cause the model to overfit the training data.

In addition, we are considering changing our KL divergence classification method completely, and using a neural network instead of KL divergence. Specifically, we will use a convolutional neural network. The input layer will match the size of the normalized SIFT feature vectors, then hidden layer will contain activation functions that will learn the building patterns in the image dataset, and the output layer will have 5 neurons, one neuron for each building in our dataset. We will use another activation function to get the probability scores for each building class.
We would  train the neural network by spliting our dataset into training and validation sets. We will use the SIFT feature vectors as inputs and the building labels as the outputs, and adjust and test to see what parameters need for fine-tuning.

To further discuss our plan going forward, we have planned a meeting with Professor Czajka to go over our KL divergence to make sure that we are implementing KL divergence and calculating our classification accuracy correctly. We can then assess if it is necessary to make very large changes with our dataset or to change our classification method to a convolutional neural network.

**Division of Labor**

1/2 Classification work done by Patrick, 1/2 Classification work done by Anna

1/2 of write-up done by Patrick, 1/2 write-up done by Anna

# Part 5 - Final Report

**Project Updates from Part 4**

_Choice of Classifier_

After meeting with Professor Czajka, we realized that using KL Divergence to classify our models was more trouble than it was worth, and was causing our project to perform very inaccurately. Instead of using KL divergence, we opted to classify our images using the scores that we received from our SIFT feature extraction. When using SIFT, we were able to identify keypoints on each image, and compare those keypoints with the other images in the dataset. We can take the score of each of these keypoint matches and calculate the mean score against images frome each building. The building corresponding to the highest keypoint mean score is the building that the image will be classified as. We decided to use this method because the mean data is a much easier metric to work with than histograms would be. In addition, the SIFT keypoint scores build very nicely on top of the pre-processing and feature extraction steps that we have already completed, whereas the KL divergence method does not nicely match the work that we have already done on our dataset.

**Description of Test Database**

Our test database is very small - 20 images total. There will be 3 "unknown" images of each building for 15 building photos total (these images were taken very recently for testing purposes, and were not used in any previous parts of the project). In addition to the building photos, we have included 10 random images to the test database that are not of any of the buildings that we have categorized, in order to test the model's reaction to images it has not been trained to recognize.

**Classification Accuracy of Test Set**

The classification accuracy of the test set is 

**Error Rate Analysis**

Most of you should see worse results on the test set when compared to the results obtained on train/validation sets. You should not be worried about that, but please provide the reasons why your solution performs worse (with a few illustrations, such as pictures or videos, what went wrong). What improvements would you propose to lower the observed error rates?

**Division of Labor**

1/2 Adjusting classification code done by Patrick, 1/2 Adjusting classification code done by Anna

1/2 of write-up done by Patrick, 1/2 write-up done by Anna





