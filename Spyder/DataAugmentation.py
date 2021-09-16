from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import util
import cv2
import random
import os


def getFileList(path):
    for a, b, file in os.walk(path):
        return file


PATH = '/Users/Linzh/Local/GitHub/uncategorized/Data/number/'
amount = 10

fileList = getFileList(PATH)
for file in fileList:
    for index in range(amount):
        img = Image.open('{}{}'.format(PATH, file))
        img = img.rotate(random.randint(-30, 30))
        img = np.array(img)
        noise_gs_img = util.random_noise(img, mode='speckle')
        noise_gs_img = noise_gs_img*255
        noise_gs_img = noise_gs_img.astype(np.int)
        # plt.show()
        cv2.imwrite('{}{}of{}.jpg'.format(PATH, file, index), noise_gs_img)
