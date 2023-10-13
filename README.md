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
