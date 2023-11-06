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

List of methods applied for pre-processing and feature extraction:
- convert colored images to grayscale images
- compute keypoints and descriptors
- match descriptors across images

The SIFT algorithm, or Scale-Invariant Feature Transform, is an algorithm that extracts distinctive features from images, despite any scale, rotation, and lighting chagnes in the images. The SIFT algorithm identifies key points within an image and computes a descriptor for each point. The descriptor can then match and recognize objects in different images.

We believed that SIFT would be the best algorithm to use for our building recognition program because it can work well despite changes in scale and rotation. Many of our images of Notre Dame buildings are taken from different angles, and thus appear to have different sizes and rotations. SIFT is able to detect and describe key features of the building despite these setbacks.

In addition, SIFT is very good at distinctive feature extraction. Many of the buildings that we photographed have very distinct architectural features (such as the Dome, the arch in Lyons Hall, and the windows in South dining hall). We wanted to leverage SIFT's ability to extract these distinct features so that we can most effectively identify these buildings.

SIFT also performs very well despite changes in lighting. Half of our data set was taken in the morning, and half was taken in the afternoon. Therefore, the lighing in the data set varies greatly. However, SIFT focuses on the structure of the image rather than the value of the pixels, so the performance of the algorithm is not greatly affected by this. 

Also, many of the building images in our data set do not capture the building entirely. Some of our images are blocked by trees and pedestrians, and in some images the photographer simply did not include the entire building in the shot. However, SIFT is very good at handling partial views, as it can match the key features that are visible even when other key features are hidden.

SIFT is also very good at creating a reference database of key features. When recognizing an unknown building, SIFT can easily generate distinctive descriptors, and match those descriptors to the most similar descriptors in the database. SIFT is also very scalable. Although the dataset is currently relatively small, if we ever wanted to expand it in the future, SIFT would make it easy to do so.

SIFT is also used in many fields that span beyond just building recognition. It can be used in more advanced cases, like facial recogntiion, scene recognition, and object tracking - its versatility is very impressive. 

Importantly, open source implementations can easily be accessed. In our case, we were able to access it using the open CV library. The open source nature of this algorithm encourages collaborative improvements, ensuring that the algorithm adapts to evolving requirements.

# Method Illustrations
Original Image:

![image](https://github.com/hsiaopat/NDBuildingRecognition/assets/97554902/50419704-5cd9-472b-b7ba-5cc0cae6704b)

Iamge after identifying key points:

![image](https://github.com/hsiaopat/NDBuildingRecognition/assets/97554902/a63bbcee-5153-46d8-b762-fd9041106543)

Original Images:
![image](https://github.com/hsiaopat/NDBuildingRecognition/assets/97554902/1f0baf12-22be-4b1a-9bac-6c0f16c537b3) ![image](https://github.com/hsiaopat/NDBuildingRecognition/assets/97554902/d6f712cd-fbe5-44f9-8d31-864a9b544a5e)





