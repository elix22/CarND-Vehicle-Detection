**Vehicle Detection Project**

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/heatmap.jpg
[image5]: ./examples/sliding_window.jpg
[video1]: ./output.avi

##Dependencies
* opencv
* scipy
* sklearn
* numpy
* anaconda (reccommended)

##Usage
run `python detect-vehicles.py`

##Feature Extraction
I used [Histogram of Oriented Gradients (HOG)](./https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) to extract features from images. The code is located in the function `single_img_features()` in file `utils.py`. Although this function supports also spatial and histogram features extraction I did not use those two in this project due to lower effectivity. 
The training data is a set of images with the size (64, 64) divided into car images and non-car images which allows easy labels generation. The code responsible for training the classifier is located in function `train()` in file `train.py`. It supports two modes of operation - reading individual images files or reading the data from archive file. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:
![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  To find the best combination of parameters I ran training and detection on one image for several combinations of those parameters and vizualized the boxes detected in each case as a car as can be seen in the following figure.
![alt text][image2]

After some trial and error I chose `orientations = 32` and `pixels_per_cell = 16` as a balance between the desire to have better separation between classes and thus lower amount of false positives and the fact that increasing the orientations parameter dramatically reduces processing speed.

##Training
Once the features are extracted from images they are scaled using `sklearn.StandardScaler` class and fed into the scaler fitting function in `train.py` using `sklearn.svc.LinearSVC` classifier. Finally near the end of `train()` function in `train.py` both the scaler and trained classifier are saved to files to enable reuse in vehicle detection algorithm.

##Sliding Window Search

There are two modes of vehicle search in the image, located in function `process_image()` in `detect-vehicles.py`:
* Full search - executed every 10 frames, searching with two different window sizes:
	* (128, 128), overlap = 0.5 - bottom 256 rows in the image, assuming that objects in this area are bigger than objects in the horizon
	* (64, 64), overlap = 0.7   - middle `[400:h-200]` rows that are closer to the horizon and cover smaller objects.
* Focused search - executed in 9 out of 10 frames and is faster than the full search. The search is executed around areas detected in previous search. For each such area two local searches are done:
	* (64, 64), overlap = 0.7 - sliding windows inside the window (expanded by 20% from each side)
	* Zoom out - searches in 10 increasingly bigger windows (each step adds 10% in each side) for each car region detected in previous frame.
The following figure provides an example of boxes detected by the full and focus searches:
![alt text][image3]

##Heatmap
Each window from the images above is passed through the SVM classifier in function `search_windows()` from the file `utils.py`. The function returns list of windows that the classifier predicted being a car. In this stage there are many false positives among those predictions and to cope with that the predictions from up to 5 frames back is converted to a heatmap. Heatmap generation is done by adding 1 to each pixel location that is inside a window, for each window detected as car and for up to 5 frames back. With this scheme hotter areas are created by overlapping windows whereas false positives don't overlap as much as real detections (hopefully).
One of the problems that surfaced from time to time is parts of a car object being "hot" - recognized as a car but other parts are not. It created islands of hot areas on the same car (when the car object is big) that were later mistaken for 2 or 3 different cars. To solve this problem I used smoothing algorithm implemented in `scipy.ndimage.morphology.grey_opening()` function in `scipy` library. This function is called after each frame's windows are added to the heatmap and effectively mitigates the effects of hot islands. Here is an example of an image from project video with the heatmap overlaid on it:
![alt text][image4]
I also experimened with "weighted" heatmap attribution, when each pixel contributes to a heatmap a value proportional to the distance of that prediction from decision boundary retrieved from the classifier. Unfortunetely this approach did not prove itself effective in practice, probably due to poor correlation of the metric to actual similarity to car.
 
##Labeling (aka Object Detection)
Given the heatmap generated from several frames, I then separate the false positives from the real cars by applying a threshold to the heatmap, regecting all pixels that are below the average of non-zero heatmap areas in function `process_image()`. The thresholded heatmap is then passed to `scipy.ndimage.measurements.label()` function that groups close pixels together and assignes them a label such as that each label represents another detected car. The final stage is to find the surrounding box for each label, which is done in function `labels_to_boxes()` in `detect-vehicles.py` by finding the max and min x and y coordinates over all the pixels with the same label. Here is the result produced by this method:
![alt text][image5]

## Video Implementation
Here's a [link to my video result](./output.avi)
![alt text][video1]

##False Positives
There are three main directions I used for false positives rejection. 
* Heatmap aggregated over several frames. This approach is described in sections above.
* Focused search - once the cars were detected in the first frame, the search can ignore other areas thus reducing false positives and increasing speed.
* Additional training data - I saved the false positives in several places in the project movie to files and added them to the "non-vehicle" directory and then retrained the classifier with those additional images. This approach was very effective in reducing the false positives.

##Discussion
False positives are still a big problem due to the trade off between speed and accuracy. One way to decrease the number of false positives is increase the number of features in each vector by increasing the number of `orientations` parameter. Unfortunetely it dramatically reduces the speed of image processing both at training and prediction stages. Profiling of the code revealed that most of the processing time is spent on extracting features, specifically executing the `skimage.hog()`, thus opportunities for optimization are limited.
Nonetheless, there are several methods that might be able to help without major hit to the speed of execution:
* Multi-sized sliding windows - current implementation has limited ability to use different window sizes however it can be generalized by correlating the window size to the object size that can be infered upon from position in the image. It can be done, for example, by applying reverse perspective tranform on the sliding windows to automatically map areas closer to the horizon to smaller window sizes.
* Multi-classifier - when the classifier is trained it uses features that were extracted using several hyper-parameters such as `orientations`, `pixels_per_cell` and type of features. However when specific set of parameters performs well on one scene, it may fail measerably on another. One solution might be to train several classifiers with different parameters and dynamically use the one that gives the best results or even combine results from several classifiers. 

