import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import numpy as np

#hog params
color_spaces = ['RGB'] #, 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']
orients = [32, 64]  # HOG orientations
pix_per_cells = [8, 16, 32]  # HOG pixels per cell
cell_per_blocks = [2]  # HOG cells per block
hog_channels = ['ALL']  # Can be 0, 1, 2, or "ALL"
#non hog params
spatial_sizes = [(32, 32)]  # Spatial binning dimensions
hist_binss = [32]  # Number of histogram bins
spatial_feat = False  # Spatial features on or off
hist_feat = False  # Histogram features on or off
hog_feat = True  # HOG features on or off
y_start_stop = [400, None]  # Min and max in y to search in slide_window()

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
    plt.tight_layout()
    plt.show()

files = glob.glob('output_images/params/*.jpg')
images = []
for file in files:
    images.append(mpimg.imread(file))
titles = ['orientations = 32, pixels_per_cell = 16',
          'orientations = 32, pixels_per_cell = 32',
          'orientations = 32, pixels_per_cell = 8',
          'orientations = 64, pixels_per_cell = 16',
          'orientations = 64, pixels_per_cell = 32',
          'orientations = 64, pixels_per_cell = 8']
show_images(images, cols=2, titles=titles)

plt.show()
exit(0)


from train import *
from detect import *

image = mpimg.imread('test_images/test1.jpg')
for color_space in color_spaces:
    for orient in orients:
        for pix_per_cell in pix_per_cells:
            for cell_per_block in cell_per_blocks:
                for hog_channel in hog_channels:
                    for spatial_size in spatial_sizes:
                        for hist_bins in hist_binss:
                            params = {'color_space': color_space, 'orient': orient,
                                      'pix_per_cell': pix_per_cell, 'cell_per_block': cell_per_block,
                                      'hog_channel': hog_channel, 'spatial_size': spatial_size,
                                      'hist_bins': hist_bins, 'spatial_feat': False,
                                      'hist_feat': False, 'hog_feat': True}
                            print(params)
                            svc, X_scaler = train(params=params)
                            processed = process_image(image, svc, X_scaler, params, video=False)
                            fname = str(color_space) + '_' + str(orient) + '_' + \
                                    str(pix_per_cell) + '_' + str(cell_per_block) + '_' + \
                                    str(hog_channel) + '_' + str(spatial_size) + '_' + \
                                    str(hist_bins)
                            fname = 'output_images/params/' + fname
                            mpimg.imsave(fname, processed)