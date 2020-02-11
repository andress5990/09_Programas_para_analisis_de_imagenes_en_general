import cv2
import numpy as np
import sys
import glob

import matplotlib.pyplot as plt
from PIL import Image

file_name = sys.argv[1]
cv_img = cv2.imread(file_name)
cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
print(cv_img.shape)
cv_img2 = cv_img[35:407, 0:500]
plt.subplot(121)
plt.imshow(cv_img)
plt.subplot(122)
plt.imshow(cv_img2)
plt.show()