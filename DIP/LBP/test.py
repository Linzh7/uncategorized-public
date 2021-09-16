import numpy as np
import pandas as pd
import os
import skimage.feature as skf
import cv2
import matplotlib.pyplot as plt
from skimage import filters
from skimage.transform import integral_image
# import testLBP as LBP


###HOG特征提取及展示###
def getHOG(image):
    # ret,thresh = cv2.threshold(fimg, 1,255,0)
    # contours,hierarchy = cv2.findContours(thresh, 3, 2)
    # if len(contours) != 0 :
    #    cnt = contours[0]
    #    for c in contours :
    #        if len(cnt) < len(c) :
    #            cnt = c
    #    x,y,w,h = cv2.boundingRect(cnt)
    #    frame= img[y:y+h,x:x+w]
    #    gesture=cv2.resize(frame, (128, 128))
    # hog = skf.hog(gesture, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    hog, hog_image = skf.hog(image, orientations=9, pixels_per_cell=(8, 8),
                             cells_per_block=(2, 2), block_norm='L1', visualize=True)
    # print(type(hog))  #为<class 'numpy.ndarray'>
    # print(np.array(hog).shape)  # (8100,)
    # print(hog)
    # print(np.max(np.array(hog)))
    hh = str(hog.tolist()).lstrip('[').rstrip(']')
    # print('hh=',hh)
    # print(type(hh))  # < class 'str' >
    temp = np.array(hog).reshape((1, len(hog)))
    print('temp=', temp)
    print(temp.shape)
    print(type(temp))
    cv2.imshow('RGBframe', img)
    cv2.imshow('hand gesture', fimg)
    cv2.imshow('gray', fimg)
    cv2.imshow('hog', hog_image)


###LBP特征提取及展示###
def getlbp(image):
    radius = 3  # LBP算法中范围半径的取值
    n_points = 8 * radius  # 领域像素点数
    lbp = skf.local_binary_pattern(image, n_points, radius, method='uniform')
    print(lbp.shape)  # np.narray(128,128)
    cv2.namedWindow('lbp', cv2.WINDOW_NORMAL)
    cv2.imshow('lbp', lbp)
    cv2.imshow('image', image)
    # print(np.max(lbp))#58
    max_bins = int(lbp.max() + 1)
    print(max_bins)
    hist, _ = np.histogram(lbp.ravel(), bins=int(
        max_bins), range=(0, max_bins))
    hist = hist.astype('float')
    # print(type(hist))
    hist /= hist.sum()
    print(hist.shape)  # (59,)
    # print(temp.shape)
    # print(type(temp))
    # print(hist.shape)
    # print(max(hist))
    # cv2.imwrite(r'rgb\bLBP.jpg',lbp)
    # plt.bar(np.arange(len(hist)), hist, width = 1, align='center')
    # plt.xlim([0.255])
    plt.show()


def calc_lbp(image, cell_size=16, hist_size=59):
    block = 128//cell_size
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
            print(img.shape)
            # lbp=skf.local_binary_pattern(img, n_points, radius, method='nri_uniform')
            lbp = skf.local_binary_pattern(img, n_points, radius)
            max_bins = int(lbp.max() + 1)
            k = i*block+j
            hist[k], _ = np.histogram(
                lbp.ravel(), bins=hist_size, range=(0, hist_size))
            hist[k] = hist[k].astype('float')
            hist[k] /= hist[k].sum()
    hist = hist.ravel()
    print(len(hist))
    plt.bar(np.arange(len(hist)), hist, width=1, align='center')
    # plt.xlim([0.255])
    plt.show()
    return hist


###Muli_block_lbp###
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
    cv2.imshow('mblbp', des)
    valid = des[cell_size:height-cell_size, cell_size:weight-cell_size].copy()
    max_bins = int(valid.max() + 1)
    hist, _ = np.histogram(valid.ravel(), bins=int(
        max_bins), range=(0, max_bins))
    hist = hist.astype('float')
    hist /= hist.sum()
    hist = hist.ravel()
    cv2.imshow('image', image)
    plt.bar(np.arange(len(hist)), hist, width=1, align='center')
    plt.show()
    return hist


def getmblbp(self, image, cell_size=2):
    a = cell_size//2
    height = image.shape[0]
    weight = image.shape[1]
    des = np.zeros((height, weight))
    for i in range(0, height-cell_size+1, cell_size):
        for j in range(0, weight-cell_size+1, cell_size):
            img = image[i:i+cell_size, j:j+cell_size].copy()
            sum = np.sum(img)

    print(des)


def getsift(image):  # 通过聚类为固定特征数目
    # gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    hand_img = cv2.resize(image, (128, 128))
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gesture, None)  # 获得关键点
    # kp, des = sift.detectAndCompute(hand_img,None)#des是是一个形状为Number_of_Keypoints×128的数字数组
    kp, des = sift.compute(hand_img, kp)
    print(des.shape)
    sift_img = cv2.drawKeypoints(
        hand_img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # 将绘制一个具有关键 点大小的圆，甚至会显示其方向
    cv2.imshow('sift', sift_img)
    cv2.imwrite('sift_keypoints.jpg', img)  # 保存特征点文件


if __name__ == '__main__':
    img = cv2.imread("./train_jpg/bfritzb2.jpg", cv2.IMREAD_COLOR)
    img2 = cv2.imread("./test_jpg/pleuchb2.jpg", cv2.IMREAD_COLOR)
    fimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gesture = cv2.resize(fimg, (128, 128))
    fimg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gesture2 = cv2.resize(fimg2, (128, 128))
    # print(gesture.shape)#(128,128)
    # print(type(gesture))#np.ndarray
    '''
    black_img=np.zeros((128,128))#lbp在59处
    write_img=np.ones((128,128))*255#lbp在59处
    # cv2.imshow('black',black_img)
    # cv2.imshow('write',write_img)
    '''
    getHOG(gesture)
    # cv2.imshow('img',gesture)
    # getlbp(gesture)
    # getuniformLBP(gesture)
    # getmblbp(gesture)
    # calc_lbp(gesture)
    # getmuiblbp(gesture2)
    # getsift(gesture)
    # plt.bar(np.arange(len(hist)), hist, width=1, align='center'）
    plt.show()
    cv2.waitKey(0)
