from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import util
import cv2
import random

font = FontProperties(fname=r"simsun.ttc", size=80)

label = 2
index = 1
amount = 200
for label in range(10):
    for index in range(amount):
        my_dpi = 20
        fig = plt.figure(figsize=(20/my_dpi, 20/my_dpi), dpi=my_dpi)
        plt.axis('off')
        plt.text(random.random()/2-0.1, random.random()/3-0.1, str(label), fontproperties=font)
        plt.savefig('{}of{}.jpg'.format(label, index))
        img = Image.open('{}of{}.jpg'.format(label, index))
        img = np.array(img)
        noise_gs_img = util.random_noise(img, mode='speckle')
        noise_gs_img = noise_gs_img*255
        noise_gs_img = noise_gs_img.astype(np.int)
        # plt.show()
        cv2.imwrite('{}of{}.jpg'.format(label, index), noise_gs_img)
