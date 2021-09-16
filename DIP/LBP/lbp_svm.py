import numpy as np
import cv2
import skimage.feature as skf
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import os
from datetime import datetime
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
# import testLBP as LBP
import pandas as pd
from skimage.transform import integral_image

GESTURE_CLASSES = ['a', 'b', 'c', 'd', 'g', 'h', 'i', 'l', 'v', 'y']
LBP_WINDOW_SIZE = 128


def getsvmmodel():
    # Train MultiClass SVM
    TRAINING_FEATURES = []
    TRAINING_LABELS = []
    trainFile = open('rgb\\rgb_lbp_training_set_' +
                     str(LBP_WINDOW_SIZE) + '.csv', 'r')
    for line in trainFile:
        print('into for')
        j = 0
        featuresVector = []
        for value in line.strip().split(','):
            if j == 0:
                TRAINING_LABELS.append(int(value))
            else:
                featuresVector.append(float(value))
            j = j+1
        TRAINING_FEATURES.append(featuresVector)
    trainFile.close()
    # print(TRAINING_LABELS)
    # print(TRAINING_FEATURES[0],TRAINING_FEATURES[1])

    # print (TRAINING_FEATURES)
    # print (TRAINING_FEATURES[0])
    print('Initializing One vs Rest Multi Class SVM from saved training data')
    print(len(TRAINING_FEATURES[0]))
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    OVR_SVM_CLF = SVC(C=100.0, kernel='rbf', gamma='auto')
    OVR_SVM_CLF.fit(TRAINING_FEATURES, TRAINING_LABELS)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Saving SVM model to lbp_svm' + str(LBP_WINDOW_SIZE) + '.pkl')
    joblib.dump(OVR_SVM_CLF, 'rgb\model\lbp_svm_rbf_' +
                str(LBP_WINDOW_SIZE) + '.pkl')


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
    block = LBP_WINDOW_SIZE//cell_size
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
            max_bins = int(lbp.max() + 1)
            k = i*block+j
            hist[k], _ = np.histogram(
                lbp.ravel(), bins=hist_size, range=(0, hist_size))
            hist[k] = hist[k].astype('float')
            hist[k] /= hist[k].sum()
    hist = hist.ravel()
    return hist


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


# Measure performance of dataset contained in some directoryL
def measurePerformance(datasetDIR):
    GESTURE_COUNT = np.zeros([len(GESTURE_CLASSES), len(GESTURE_CLASSES)])
    print('================================================================================================')
    print(datasetDIR)
    OVR_SVM_CLF = joblib.load(
        './rgb/model/lbp_svm_rbf_' + str(LBP_WINDOW_SIZE) + '.pkl')
    # Process files in directory
    for root, dirs, files in os.walk(datasetDIR):
        for file_name in files:
            if file_name.split('.')[0][-1] in ['1']:
                handgestureFile = os.path.join(
                    root, file_name).replace('\\', '/')
                tag = file_name.split('.')[0][-2]
                print('Processing ' + handgestureFile)

                # Extract features per frame
                frame = cv2.imread(handgestureFile)

                # try :
                graySkin = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                graySkin = cv2.GaussianBlur(graySkin, (3, 3), 0)
                # Find contours
                #ret,thresh = cv2.threshold(graySkin, 1,255,0)
                #contours,hierarchy = cv2.findContours(thresh, 1, 2)
                #
                #
                # if len(contours) != 0 :
                #    cnt = contours[0]
                #    for c in contours :
                #        if len(cnt) < len(c) :
                #            cnt = c
                #    x,y,w,h = cv2.boundingRect(cnt)
                #    segmentedGraySkin = graySkin[y:y+h,x:x+w]
                #
                #    # Compute HOG
                resizedSegmentedGraySkin = cv2.resize(
                    graySkin, (LBP_WINDOW_SIZE, LBP_WINDOW_SIZE))
                hist = getmuiblbp(resizedSegmentedGraySkin)
                temp = np.array(hist).reshape((1, len(hist)))
                prediction = OVR_SVM_CLF.predict(temp)[0]
                print(tag, prediction)
                GESTURE_COUNT[prediction][GESTURE_CLASSES.index(
                    tag)] = GESTURE_COUNT[prediction][GESTURE_CLASSES.index(tag)] + 1

                # except:
                #print ('An error occurred')

    # Print results; exclude background class
    print(GESTURE_COUNT)
    TOTAL_PER_CLASS = sum(GESTURE_COUNT)
    TOTAL_CORRECT = 0
    i = 0
    while i < len(GESTURE_CLASSES):
        print(GESTURE_CLASSES[i] + ' = ' + str(GESTURE_COUNT[i][i]) + '/' + str(
            TOTAL_PER_CLASS[i]) + '(' + str(round(GESTURE_COUNT[i][i]*100.0/TOTAL_PER_CLASS[i], 2)) + '%)')
        TOTAL_CORRECT = TOTAL_CORRECT + GESTURE_COUNT[i][i]
        i = i+1
    print('')
    print('OVERALL = ' + str(TOTAL_CORRECT) + '/' + str(sum(TOTAL_PER_CLASS)) +
          '(' + str(round(TOTAL_CORRECT*100.0/sum(TOTAL_PER_CLASS), 2)) + '%)')


if __name__ == "__main__":
    getsvmmodel()
    measurePerformance('rgb\\train_jpg12')
    measurePerformance('rgb\\test_jpg12')
    cv2.destroyAllWindows()
