import cv2
import numpy as np
import sys

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import style
style.use('fivethirtyeight')

import pdb

file_name = sys.argv[1] 
img = cv2.imread(file_name)

#plt.imshow(img, cmap='brg')
#plt.show()

#cv2.namedWindow('Imagen', cv2.WINDOW_AUTOSIZE)
#cv2.imshow('Imagen', img)
cv2.waitKey()
