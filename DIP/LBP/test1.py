import matplotlib.pyplot as plt
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVR
from skimage import feature as skft
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data, io, data_dir, filters, feature
from skimage.color import label2rgb
import skimage
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import csv
# 常数
train_amount = 511
test_amount = 207
# lbp常量
radius = 1.0
n_point = radius * 8


def loadPicture():
    # read in
    with open('train_jpg/list.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    train_name_data = np.array(rows)
    with open('test_jpg/list.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    test_name_data = np.array(rows)
    # init data
    train_data = np.zeros((train_amount, 128, 128))
    test_data = np.zeros((test_amount, 128, 128))
    train_label = np.zeros((train_amount))
    test_label = np.zeros((test_amount))
    # init lable
    count = 0
    tmp = train_name_data[0][1]
    for i in range(train_amount):
        train_data[i] = cv2.imread(
            "train_jpg/" + train_name_data[i][0] + ".jpg")
        if tmp == train_name_data[i][1]:
            train_label[i] = count
        else:
            count += 1
            tmp = train_name_data[i][1]
            train_label[i] = count
    # init
    count = 0
    tmp = test_name_data[0][1]
    for i in range(test_amount):
        test_data[i] = cv2.imread("test_jpg/" + test_name_data[i][0] + ".jpg")
        if tmp == test_name_data[i][1]:
            test_label[i] = count
        else:
            count += 1
            tmp = test_name_data[i][1]
            test_label[i] = count
    return train_data, test_data, train_label, test_label


def texture_detect():
    train_hist = np.zeros((train_amount, 256))
    test_hist = np.zeros((test_amount, 256))
    for i in np.arange(train_amount):
        # 使用LBP方法提取图像的纹理特征.
        lbp = skft.local_binary_pattern(
            train_data[i], n_point, radius, method='default')
        # 统计图像的直方图
        max_bins = int(lbp.max() + 1)
        # hist size:256
        train_hist[i], _ = np.histogram(lbp, 256, [0, 256])
    for i in np.arange(test_amount):
        lbp = skft.local_binary_pattern(
            test_data[i], n_point, radius, method='default')
        plt.imshow(lbp)
        plt.show()
        # 统计图像的直方图
        max_bins = int(lbp.max() + 1)
        # hist size:256
        test_hist[i], _ = np.histogram(
            lbp, normed=True, bins=max_bins, range=(0, max_bins))
    print('train\n', train_hist)
    print('test\n', test_hist)
    return train_hist, test_hist


train_data, test_data, train_label, test_label = loadPicture()
train_hist, test_hist = texture_detect()
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
OneVsRestClassifier(svr_rbf, -1).fit(train_hist,
                                     train_label).score(test_hist, test_label)
