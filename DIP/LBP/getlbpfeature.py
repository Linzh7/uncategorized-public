import numpy as np
import cv2
import os
import skimage.feature as skf
# import testLBP as LBP
from skimage.transform import integral_image

GESTURE_CLASSES = ['a', 'b', 'c', 'd', 'g', 'h', 'i', 'l', 'v', 'y']
DIR = 'rgb\\train_jpg'
LBP_WINDOW_SIZE = 128


# For HOG Feature
LBP_FEATURES_FILE = 'rgb\\rgb_lbp_training_set_' + \
    str(LBP_WINDOW_SIZE) + '.csv'
lbpFeaturesFile = open(LBP_FEATURES_FILE, "w")


def getlbp(image):
    radius = 2  # LBP算法中范围半径的取值
    n_points = 8 * radius  # 领域像素点数
    eps = 1e-7
    lbp = skf.local_binary_pattern(
        image, n_points, radius, method='nri_uniform')
    max_bins = int(np.max(lbp) + 1)
    hist, _ = np.histogram(lbp, bins=int(max_bins), range=(0, max_bins))
    hist = hist.astype('float')
    hist /= (hist.sum() + eps)
    return hist


def calc_lbp(image, cell_size=16, hist_size=59):
    block = LBP_WINDOW_SIZE // cell_size
    hist = np.zeros((block*block, hist_size))
    radius = 1  # LBP算法中范围半径的取值
    n_points = 8 * radius  # 领域像素点数
    for i in range(block):
        for j in range(block):
            a = i*cell_size
            b = (i+1)*cell_size
            c = j*cell_size
            d = (j+1)*cell_size
            img = image[a:b, c:d]
            lbp = skf.local_binary_pattern(
                img, n_points, radius, method='nri_uniform')
            #lbp=skf.local_binary_pattern(img, n_points, radius)
            k = i*block+j
            hist[k], _ = np.histogram(
                lbp.ravel(), bins=hist_size, range=(0, hist_size))
            hist[k] = hist[k].astype('float')
            hist[k] /= hist[k].sum()
    hist = hist.ravel()
    return hist


'''
def getmuiblbp(image, cell_size=8):
    hist=np.zeros((1,128*128))
    for i in range(128):
        for j in range(128):
            int_img = integral_image(image)
            mblbp=skf.multiblock_lbp(int_img,i,j,cell_size,cell_size)
            k=i*128+j
            hist[0][k] = mblbp
            hist[0][k] = hist[0][k].astype('float')
    hist[0] /= 256 
    hist=hist.ravel()
    #cv2.imshow('image',image)
    #plt.bar(np.arange(len(hist)), hist, width = 1, align='center')
    #plt.show()
    return hist
'''


def getmuiblbp(image, cell_size=2):
    a = cell_size//2
    height = image.shape[0]
    weight = image.shape[1]
    des = np.zeros((height, weight))
    int_img = integral_image(image)
    for i in range(0, height-2*cell_size, cell_size):
        for j in range(0, weight-2*cell_size, cell_size):
            mblbp = skf.multiblock_lbp(int_img, i, j, cell_size, cell_size)
            for m in range(cell_size):
                for n in range(cell_size):
                    des[i + cell_size + m][j + cell_size + n] = mblbp
    valid = des[cell_size:height-cell_size, cell_size:weight-cell_size].copy()
    max_bins = int(valid.max() + 1)
    hist, _ = np.histogram(valid.ravel(), bins=int(
        max_bins), range=(0, max_bins))
    hist = hist.astype('float')
    hist /= hist.sum()
    hist = hist.ravel()
    return hist


# Process files in directory
for root, dirs, files in os.walk(DIR):
    for file_name in files:
        if file_name.split('.')[0][-1] in ['1']:
            handgestureFile = os.path.join(root, file_name).replace('\\', '/')
            tag = file_name.split('.')[0][-2]
            print('Processing ' + handgestureFile)

            # Extract features per frame
            frame = cv2.imread(handgestureFile)
            # try :
            graySkin = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #smallerGray = cv2.resize(graySkin, (HOG_WINDOW_SIZE, HOG_WINDOW_SIZE))
            # Find contours
            #ret,thresh = cv2.threshold(graySkin, 1,255,0)
            #contours,hierarchy = cv2.findContours(thresh, 1, 2)
            #
            # if len(contours) != 0 :
            #    cnt = contours[0]
            #    for c in contours :
            #        if len(cnt) < len(c) :
            #            cnt = c
            #    x,y,w,h = cv2.boundingRect(cnt)
            #
            #    # Compute HOG
            #    segmentedGraySkin = graySkin[y:y+h,x:x+w]
            #    resizedSegmentedGraySkin = cv2.resize(segmentedGraySkin, (HOG_WINDOW_SIZE, HOG_WINDOW_SIZE))
            resizedSegmentedGraySkin = cv2.resize(
                graySkin, (LBP_WINDOW_SIZE, LBP_WINDOW_SIZE))
            hist = getmuiblbp(resizedSegmentedGraySkin)

            if not np.max(hist):
                print('lbp is empty')
            else:
                lbpFeaturesFile.write(str(GESTURE_CLASSES.index(
                    tag)) + ',' + str(hist.tolist()).lstrip('[').rstrip(']') + '\n')
                # print(tag)

plt.bar(np.arange(len(hist)), hist, width=1, align='center')
plt.show()

cv2.destroyAllWindows()
lbpFeaturesFile.close()
