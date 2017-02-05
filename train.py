#train a classifier and save to file:
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from utils import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

from_archive = False

def train(params):
    if params is not None:
        color_space = params['color_space'] #'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        orient = params['orient'] #32  # HOG orientations
        pix_per_cell = params['pix_per_cell'] #16  # HOG pixels per cell
        cell_per_block = params['cell_per_block'] #2  # HOG cells per block
        hog_channel = params['hog_channel'] #0  # Can be 0, 1, 2, or "ALL"
        spatial_size = params['spatial_size'] #(32, 32)  # Spatial binning dimensions
        hist_bins = params['hist_bins'] #16  # Number of histogram bins
        spatial_feat = params['spatial_feat']#False  # Spatial features on or off
        hist_feat = params['hist_feat']#True  # Histogram features on or off
        hog_feat = params['hog_feat']#True  # HOG features on or off

    car_imgs = []
    notcar_imgs = []
    #read all the files:
    if not from_archive:
        print('Enumerating images')
        # Read in cars and notcars
        cars = []
        notcars = []
        cars += glob.glob('dataset/vehicles/GTI_Right/*.png')
        cars += glob.glob('dataset/vehicles/GTI_Left/*.png')
        cars += glob.glob('dataset/vehicles/GTI_MiddleClose/*.png')
        cars += glob.glob('dataset/vehicles/GTI_Far/*.png')
        cars += glob.glob('dataset/vehicles/KITTI_extracted/*.png')
        cars += glob.glob('dataset/vehicles/Extracted/Black/*.png')
        notcars += glob.glob('dataset/non-vehicles/GTI/*.png')
        notcars += glob.glob('dataset/non-vehicles/Extras/*.png')
        notcars += glob.glob('dataset/non-vehicles/Extracted/f0/*.png')
        notcars += glob.glob('dataset/non-vehicles/Extracted/f320/*.png')
        notcars += glob.glob('dataset/non-vehicles/Extracted/f470/*.png')
        notcars += glob.glob('dataset/non-vehicles/Extracted/f600/*.png')
        notcars += glob.glob('dataset/non-vehicles/Extracted/f1000/*.png')

        for i in trange(len(cars)):
            file = cars[i]
            # Read in each one by one
            image = mpimg.imread(file)
            if image.shape[:2] != (64, 64):
                continue
            car_imgs.append(image[:,:,:3])
        for i in trange(len(notcars)):
            file = notcars[i]
            # Read in each one by one
            image = mpimg.imread(file)
            if image.shape[:2] != (64, 64):
                continue
            #print(image.shape)
            notcar_imgs.append(image[:,:,:3])
        #save dataset to file:
        fname = 'dataset.npz'
        print('Saving to archive ', fname)
        np.savez(fname, cars=car_imgs, notcars=notcar_imgs)
    else:
        print('Reading dataset file')
        dataset = np.load('dataset.npz')
        car_imgs = dataset['cars']
        notcar_imgs = dataset['notcars']

    print('cars, noncars', len(cars), len(notcars))

    print('Extracting features from car images')
    car_features = extract_features(car_imgs, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)

    print('Extracting features from non-car images')
    notcar_features = extract_features(notcar_imgs, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

    print('Scaling data')
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = 0#np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.1, random_state=rand_state)

    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    print('Training classifier')
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

    #save to file
    import pickle
    s = pickle.dumps(svc)
    text_file = open("trained_classifier.pkl", "w")
    text_file.write(s)
    text_file.close()
    np.savez('scaler.npz', mean=X_scaler.mean_, scale=X_scaler.scale_)
    return svc, X_scaler


if __name__ == '__main__':
    # Hog params
    color_space = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 32  # HOG orientations
    pix_per_cell = 16  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"

    spatial_size = (32, 32)  # Spatial binning dimensions
    hist_bins = 1024  # Number of histogram bins
    spatial_feat = False  # Spatial features on or off
    hist_feat = False  # Histogram features on or off
    hog_feat = True  # HOG features on or off

    params = {'color_space': color_space, 'orient': orient,
              'pix_per_cell': pix_per_cell, 'cell_per_block': cell_per_block,
              'hog_channel': hog_channel, 'spatial_size': spatial_size,
              'hist_bins': hist_bins, 'spatial_feat':spatial_feat,
              'hist_feat':hist_feat, 'hog_feat':hog_feat}


    train(params=params)