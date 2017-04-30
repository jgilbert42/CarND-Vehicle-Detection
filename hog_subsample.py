import time
import matplotlib.image as mpimg
import numpy as np
import pickle
import cv2
from glob import glob
from scipy.ndimage.measurements import label
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from moviepy.editor import VideoFileClip
from lesson_functions import *

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, return_features=False):
    result = []

    if np.max(img) > 1:
        img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    # make sure there's at least 1 step for the return features case of window and size the same
    nxsteps = max(1, (nxblocks - nblocks_per_window) // cells_per_step)
    nysteps = max(1, (nyblocks - nblocks_per_window) // cells_per_step)
    #print(img.shape, ch1.shape)
    #print('foo', nxblocks, nyblocks, nfeat_per_block, nblocks_per_window, cells_per_step, pix_per_cell)
    #7, 7, 36, 7, 2, 8
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    #print ('steps', nxsteps, nysteps)
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins, bins_range=(0,1))

            #spatial_features = []
            #hist_features = []

            # Scale features and make a prediction
            img_features = np.hstack((spatial_features, hist_features, hog_features))
            if return_features:
                result.append(img_features)
            else:
                # not sure what reshape is for, but I guess transform needs a list
                img_features = img_features.reshape(1, -1)
                test_features = X_scaler.transform(img_features)
                test_prediction = svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    result.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart))) 

    return result

orient = 9
pix_per_cell = 8
cell_per_block = 2
spatial_size = (32, 32)
hist_bins = 32

def extract_features(img_file):
    img = mpimg.imread(img_file)
    features = find_cars(img, 0, img.shape[0], 1, None, None, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, True)
    return features[0]

svc = None
X_scaler = None

model_file = 'model.p'

try:
    data = pickle.load(open(model_file, "rb"))
    svc = data["svc"]
    X_scaler = data["X_scaler"]
    print("Using saved model")
except FileNotFoundError:
    cars = glob('data/vehicles/*/*.png')
    notcars = glob('data/non-vehicles/*/*.png')

    #sample_size = 1000
    #cars = cars[0:sample_size]
    #notcars = notcars[0:sample_size]

    car_features = []
    notcar_features = []

    print('start extracting training features')
    t = time.time()
    for f in cars:
        car_features.append(extract_features(f))

    for f in notcars:
        notcar_features.append(extract_features(f))
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract training features')

    svc = LinearSVC()
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    # train
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

    t = time.time()
    print("starting training")
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')

    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    print('pickling model')
    pickle.dump({"svc": svc, "X_scaler": X_scaler}, open(model_file, "wb"))

ystart = 400
ystop = 656
#scale = 1.5
#scale = 2
#scale = 1

# collects boxes for each frame to generate heatmaps over several frames
# TODO: currently grows without bound
heat_boxes = []

def process_image(img, return_heat=False):
    img_boxes = []
    for scale in (1, 1.5, 2, 2.5):
        boxes = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        img_boxes = img_boxes + boxes

    heat_boxes.append(img_boxes)
    heat = heatmap(img, sum(heat_boxes[-10:], []), 10)
    labels = label(heat)

    draw_labeled_boxes(img, labels)
    if return_heat:
        return img, heat
    else:
        return img

def process_video(in_file, out_file):
    clip = VideoFileClip(in_file)
    video_clip = clip.fl_image(process_image)
    video_clip.write_videofile(out_file, audio=False)

def test_images():
    for f in glob('test_images/*.jpg'):
        img = mpimg.imread(f)

        img, heat = process_image(img, True)

        show_before_after(img, heat, 'hot')

#test_images()
#process_video('test_video.mp4', 'output_test_video.mp4')
process_video('project_video.mp4', 'output_project_video.mp4')
