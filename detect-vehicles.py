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
from scipy.ndimage.measurements import *
from scipy.ndimage.morphology import grey_opening, grey_closing, grey_dilation, grey_erosion

from train import *

def list_intersection(l1, l2):
    return list(set(l1) & set(l2))

history = []
detections = []
detections_strength = []

import math

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1]-p2[1]) ** 2)
def box_size(box):
    return dist(box[0], box[1])
def box_center(box):
    return ((box[0][0] + box[1][0])/2, (box[0][1] + box[1][1]) / 2)
def box_dist(box1, box2):
    center1 = box_center(box1)
    center2 = box_center(box2)
    return dist(center1, center2)

win_size = 64

#returns true if the box is close to one of the boxes in 'boxes'
def similar(boxes, box, size_thresh=2, dist_thres=1.5*win_size):
    for box1 in boxes:
        size1 = box_size(box)
        size2 = box_size(box1)
        max_size = max(size1, size2)
        min_size = min(size1, size2)
        if max_size/min_size < size_thresh and box_dist(box1, box) < dist_thres:
            return True
    return False

def update_detections(boxes, lookback=5):

    print('boxes', boxes)
    detected = []
    if len(detections) > lookback:
        for box in boxes:
            fullhist = True
            for i in range(lookback):
                prev_boxes = detections[-i-1]
                print('prev_boxes', prev_boxes)
                if not similar(prev_boxes, box):
                    fullhist = False
                    break

            if fullhist:
                detected.append(box)
        print('detected', detected)

    detections.append(boxes)
    return detected


def enclosing_box(windows):
    winarr = np.array(windows)
    xmin = winarr[:,0,0].min()
    xmax = winarr[:,1,0].max()
    ymin = winarr[:,0,1].min()
    ymax = winarr[:,1,1].max()
    return ((xmin, ymin), (xmax, ymax))


def calc_clusters(windows, min_samples=1):
    from sklearn.cluster import KMeans
    from sklearn.cluster import DBSCAN

    if len(windows) < min_samples:
        return []
    X = np.array(windows)[:,0]
    #kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    db = DBSCAN(eps=1.5*win_size, min_samples=1).fit(X)
    print(db.labels_)
    clusters = []
    for i in range(len(db.labels_)):
        label = db.labels_[i]
        if len(clusters) > label:
            clusters[label].append(windows[i])
        else:
            clusters.append([windows[i]])

    return clusters


def add_heat(heatmap, bbox_list, strengths):
    # Iterate through list of bboxes
    for i in range(len(bbox_list)):
        box = bbox_list[i]
        strength = int(abs(strengths[i]))
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1#strength

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap < threshold] = 0
    # Return thresholded map
    return heatmap

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

#from http://stackoverflow.com/questions/22314949/compare-two-bounding-boxes-with-each-other-matlab
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0][0], boxB[0][0])
    yA = max(boxA[0][1], boxB[0][1])
    xB = min(boxA[1][0], boxB[1][0])
    yB = min(boxA[1][1], boxB[1][1])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[0][0] - boxA[1][0] + 1) * (boxA[0][1] - boxA[1][1] + 1)
    boxBArea = (boxB[0][0] - boxB[1][0] + 1) * (boxB[0][1] - boxB[1][1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def remove_overlapping(windows):
    ret = []
    for i in range(len(windows)):
        for j in range(i+1, len(windows)):
            print('iou', bb_intersection_over_union(windows[i], windows[j]))

def labels_to_boxes(labels):
    boxes = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        boxes.append(bbox)
    # return the boxes
    return boxes


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

#from: https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

def save_windows(image, windows, size=(64,64)):
    #extract window
    for window in windows:
#        if window[0][0] > 800:
#            continue
        subimg = image[window[0][1]:window[1][1], window[0][0]:window[1][0], :]
        subimg = cv2.resize(subimg, size)
        fname = 'output_images/extracted/' + 'f' + str(frame_cnt) + '_' + str(window) + '.png'
        plt.imsave(fname, subimg)


# Define a function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(50, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    abs_sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    # 5) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 6) Create a binary mask where mag thresholds are met
    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # 7) Return this mask as your binary_output image
    return sobel_binary

def show_histogram(image):
    bin_img = mag_thresh(image)
    #print(len(bin_img[bin_img>0]))
    plt.imshow(bin_img*255, cmap='gray')
    hist = np.sum(bin_img, axis=0)
    hist = smooth(hist)
    from scipy.signal import find_peaks_cwt
    indexes = find_peaks_cwt(hist, np.arange(1, 550))
    print(indexes)
    #plt.figure()
    #plt.plot(hist)
    #plt.show()
    return hist, indexes


def overlay_heatmap(image, heatmap):
    heatmap_cpy = np.copy(7 * heatmap)
    heatmap_cpy[heatmap_cpy > 255] = 255
    heatmap_cpy = cv2.cvtColor(heatmap_cpy, cv2.COLOR_GRAY2BGR).astype('uint8')
    heatmap_cpy = cv2.applyColorMap(heatmap_cpy, cv2.COLORMAP_HOT).astype('uint8')
    heatmap_cpy = cv2.cvtColor(heatmap_cpy, cv2.COLOR_BGR2RGB).astype('uint8')
    img_cpy = np.copy(image)
    img_cpy = cv2.cvtColor(img_cpy, cv2.COLOR_RGB2HSV)
    img_cpy[:, :, 2] /= 2
    img_cpy[:, :, 1] /= 2
    img_cpy = cv2.cvtColor(img_cpy, cv2.COLOR_HSV2RGB)
    # print(heatmap.shape)
    img_cpy[heatmap_cpy != 0] = heatmap_cpy[heatmap_cpy != 0]
    return img_cpy

scan_windows_images = []
def process_image(image, svc, X_scaler, params=None, video=True):
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

    #image = mpimg.imread('test_images/test1.jpg')
    draw_image = np.copy(image)
    h = image.shape[0]
    w = image.shape[1]

    windows = []

    if len(detections) % 10 == 0:
        windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[h-256, None],
                               xy_window=(128, 128), xy_overlap=(0.5, 0.5))
        windows += slide_window(image, x_start_stop=[None, None], y_start_stop=[400, h-200],
                                    xy_window=(64, 64), xy_overlap=(0.7, 0.7))
        #scan_windows_images.append(draw_boxes(np.copy(image), windows, color=(255, 0, 255), thick=2))
    else:
        for window in detections[-1]:
            slack_ratio = 0.2
            xsize = abs(window[0][0] - window[1][0])
            ysize = abs(window[0][1] - window[1][1])
            xstart, xend = max(int(window[0][0] - slack_ratio*xsize), 0), min(int(window[1][0] + slack_ratio*xsize), w)
            ystart, yend = max(int(window[0][1] - slack_ratio*ysize), 0), min(int(window[1][1] + slack_ratio*ysize), h)
            windows += slide_window(image, x_start_stop=[xstart, xend], y_start_stop=[ystart, yend],
                                    xy_window=(64, 64), xy_overlap=(0.7, 0.7))
            windows += zoom_window(image, window, limit=10, step=(0.2,0.2))
        #scan_windows_images.append(draw_boxes(np.copy(image), windows, color=(255, 0, 255), thick=2))

    #if len(scan_windows_images) >= 2:
    #    show_images(scan_windows_images, titles=['Full Search Windows', 'Focused Search Windows'])


    hot_windows, conf = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)


    '''
    windows_tosave = slide_window(image, x_start_stop=[1000, None], y_start_stop=[400, 550],
                           xy_window=(200, 200), xy_overlap=(0.5, 0.5))
    save_windows(image, windows_tosave)
    '''

    s = np.ones((3, 3))
    if video:
        history.append(hot_windows)
        detections_strength.append(conf)
        heatmap = np.zeros_like(image[:,:,0])
        lookback = 5
        for i in range(min(len(history),lookback)):
            prev_windows = history[-i-1]
            prev_strengths = detections_strength[-i-1]
            heatmap = add_heat(heatmap, prev_windows, prev_strengths)
            heatmap = grey_opening(heatmap, size=(15,15))


        #print('median', np.mean(heatmap[heatmap>0]))
        avg_density = np.mean(heatmap[heatmap>0])
        heatmap_thresh = apply_threshold(np.copy(heatmap), avg_density-1)

        labels = label(heatmap_thresh, structure=s)
        print(labels[1], 'cars found')


        detected_boxes = labels_to_boxes(labels)
        remove_overlapping(detected_boxes)
        detections.append(detected_boxes)
        print('detected', detected_boxes)
        # Draw bounding boxes on a copy of the image
        draw_img = draw_labeled_bboxes(np.copy(image), labels)

    if visualize:
        heatmap_img = overlay_heatmap(image, heatmap)
        heatmap_thresh_img = overlay_heatmap(image, heatmap_thresh)
        show_images([heatmap_img, ], titles=['Heatmap', 'Thresholded Heatmap'])


    #visualize heatmap
    if heatmap.max() > 0:
        #normalize
        #heatmap = 255.*heatmap/float(heatmap.max())
        heatmap = 5*heatmap
        heatmap[heatmap > 255] = 255
        heatmap = cv2.resize(heatmap, (w/5, h/5)).astype('uint8')
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR).astype('uint8')
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT).astype('uint8')
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype('uint8')
        small_img = cv2.resize(image, (w/5, h/5))
        small_img = cv2.cvtColor(small_img, cv2.COLOR_RGB2HSV)
        small_img[:,:,2] /= 2
        small_img[:,:,1] /= 2
        small_img = cv2.cvtColor(small_img, cv2.COLOR_HSV2RGB)
        #print(heatmap.shape)
        small_img[heatmap != 0] = heatmap[heatmap != 0]
        txtsize = 1
        thickness = 1
        txtcolor = (255, 0, 0)  #Blue
        cv2.putText(small_img, 'Heatmap', (60, 30), cv2.FONT_HERSHEY_DUPLEX, txtsize, txtcolor, thickness)
        draw_img[:h/5,:w/5,:] = small_img
        #draw_img = draw_boxes(draw_img, hot_windows, color=(0, 0, 255), thick=2)

    return draw_img


if __name__ == '__main__':
    import pickle

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

    text_file = open("trained_classifier.pkl", "r")
    s = text_file.read()
    svc = pickle.loads(s)
    text_file.close()

    scaler_params = np.load('scaler.npz')
    X_scaler = StandardScaler()
    X_scaler.mean_ = scaler_params['mean']
    X_scaler.scale_ = scaler_params['scale']
    print('scaler', X_scaler.mean_, X_scaler.scale_)

    cap = cv2.VideoCapture("project_video.mp4")
    #cap = cv2.VideoCapture("challenge_video.mp4")
    #cap = cv2.VideoCapture("harder_challenge_video.mp4")

    # Define the codec and create VideoWriter object
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #fourcc = cv2.VideoWriter_fourcc(*'X264')

    out = None

    frame_cnt = 0
    frame_start = 0
    frame_end = 0xffffffff
    visualize = False
    images = []
    while True:
        flag, image = cap.read()
        if flag:
            frame_cnt += 1
            if frame_cnt < frame_start:
                continue
            elif frame_cnt > frame_end:
                break
            print('frame_cnt = ', frame_cnt)
            if out is None:
                out = cv2.VideoWriter('output_temp.avi', fourcc, 20.0, (image.shape[1], image.shape[0]))

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            res = process_image(image, svc, X_scaler, params=params)
            res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
            cv2.imshow('video', res)
            #images.append(res)
            #if len(images) > 10:
            #    show_images(images[-4:], cols=2)
            out.write(res)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

        import time

    time.sleep(5)  # delays for 5 seconds

    # Release everything if job is finished
    if out is not None:
        out.release()
exit(0)




######################
# SCRATCHPAD
######################
def scratchpad():
    #heatmap = np.zeros_like(image[:, :, 0])
    #heatmap = add_heat(heatmap, hot_windows, conf)

    #s = np.ones((3,3))
    #labels = label(heatmap,structure=s)
    #hot_windows = labels_to_boxes(labels)
    #print('first pass', hot_windows)
    '''
    #Second pass - refined window sizes in attention regions
    small_hot_windows = []
    for window in hot_windows:
        xsize = abs(window[0][0] - window[1][0])
        ysize = abs(window[0][1] - window[1][1])
        slack_ratio = 0.2
        #xstart, xend = max(int(window[0][0] - slack_ratio*xsize), 0), min(int(window[1][0] + slack_ratio*xsize), w)
        #ystart, yend = max(int(window[0][1] - slack_ratio*ysize), 0), min(int(window[1][1] + slack_ratio*ysize), h)
        xstart, xend = max(int(window[0][0] - win_size), 0), min(int(window[1][0] + win_size), w)
        ystart, yend = max(int(window[0][1] - win_size), 0), min(int(window[1][1] + win_size), h)
        print('start, end', xstart, xend, ystart, yend)

        windows = slide_window(image, x_start_stop=[xstart, xend], y_start_stop=[ystart, yend],
                                xy_window=(64, 64), xy_overlap=(0.7, 0.7))
        small_windows, small_conf = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                                     spatial_size=spatial_size, hist_bins=hist_bins,
                                     orient=orient, pix_per_cell=pix_per_cell,
                                     cell_per_block=cell_per_block,
                                     hog_channel=hog_channel, spatial_feat=spatial_feat,
                                     hist_feat=hist_feat, hog_feat=hog_feat)
        small_hot_windows += small_windows
        conf += small_conf

    hot_windows += small_hot_windows
    #save_windows(image, hot_windows)
    '''


    '''
    hist, peaks = show_histogram(image)
    for peak in peaks:
        peak = min(peak, w)
        xstart, xend = max(peak-128, 0), min(peak+128, w)
        print(xstart, xend)
        #ystart, yend = max(int(window[0][1] - win_size), 0), min(int(window[1][1] + win_size), h)
        windows += slide_window(image, x_start_stop=[xstart, xend], y_start_stop=[400, None],
                                xy_window=(64, 64), xy_overlap=(0.6, 0.6))
    '''
