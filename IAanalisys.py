import cv2
import numpy as np
from numpy import inf
import sys
import math

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
#style.use('fivethirtyeight')

from PIL import Image

import pdb


def cal_luminosity(file_name):
    
    cv_img = cv2.imread(file_name)#np.array con shape(filas, columnas, canales)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) #volvemos el valor del np.array a RGB para procesar
    cv_img = cv_img[40:407, 0:500]
    #cv_img = cv2.GaussianBlur(cv_img, (5,5), 0)
    #b, g, r = cv2.split(cv_img)#separamos los canales en gbr
    b = cv_img[:,:,0]#Estas lineas hacen lo mismo que la anterior, pero mas eficiente
    g = cv_img[:,:,1]
    r = cv_img[:,:,2] 
    
    #Este for hace lo mismo que cv_img2 = cv2.imread(file_name, 0)
    holder = np.ones(cv_img[:,:,0].shape)
    for i in range(b.shape[0]):#Filas
        for j in range(b.shape[1]):#Columnas
            lum=0.2126*r[i][j] + 0.7152*g[i][j] + 0.0722*b[i][j] #Photometric/digital ITU BT.709
            #lum=0.299*r[i][j] +0.587*g[i][j]+0.114*b[i][j] #Digital ITU BT.601 
                                                            #(gives more weight to the R and B components)
            holder[i][j] = lum
    
    #Normalizamos la matriz
    
    maxholder = np.amax(holder)
    for i in range(b.shape[0]):
        for j in range(b.shape[1]): 
            holder[i][j] = holder[i][j]/maxholder
    
    return(holder, maxholder)

def cal_ratio(img1, img2, row, column, cv_img1max, cv_img2max):
 
    holder = np.ones((row, column))
    #pdb.set_trace()
    for i in range(holder.shape[0]):
        for j in range(holder.shape[1]):
            if img1[i][j]/img2[i][j] == np.inf :
                holder[i][j]= 0
            elif img1[i][j]/img2[i][j] != np.inf:
                holder[i][j] = img1[i][j]/img2[i][j]
            
    NaN = np.isnan(holder)#Preguntamos donde estan los nan
    holder[NaN] = 0#Reemplazamos los NaN por ceros

    return holder

def reshape(img1):
    long = img1.shape[0]* img1.shape[1]
    holder = np.ones((long))
    for i in range(img1.shape[0]):#fila
        for j in range(img1.shape[0]):#columna
            holder[j] = img1[i][j]

    return holder
    
#--------------------------------------------------------------------------------------------------            

images_names = [sys.argv[1], sys.argv[2]]
#pdb.set_trace()

#creamos ambas matrices a partir de del calculo de luminosidad (grayscale)
cv_img1 = cal_luminosity(images_names[0])[0]
cv_img1max = cal_luminosity(images_names[0])[1]#Son de las imagenes no normalizadas
cv_img2 = cal_luminosity(images_names[1])[0]
cv_img2max = cal_luminosity(images_names[1])[1]#Son de las imagenes no normalizadas


#Creamos la imagen con la razón de las otras dos
row = len(cv_img1[:,1])
column = len(cv_img1[1,:])
cv_result = cal_ratio(cv_img1, cv_img2, row, column,
                      cv_img1max, cv_img2max)

cv_img1_11 = reshape(cv_img1)
cv_img2_11 = reshape(cv_img2)

#pdb.set_trace()

cv_result_min = np.nanmin(cv_result)
cv_result_max = np.nanmax(cv_result)
bounds = np.linspace(cv_result_min, cv_result_max, 10, endpoint=True)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
pltimg = ax1.imshow(cv_result, cmap = plt.cm.jet)
plt.title('ratios of intensity values between ' + str(sys.argv[1]) + "/" + str(sys.argv[2])) #El ponemos el titulo a la grafica
plt.ylabel('pixels y coordinate')#ponemos la etiqueta de eje y
plt.xlabel('pixels x coordinate')#ponemos la etiqueta de eje x
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(pltimg, cax = cax, ticks = bounds)
#plt.colorbar(pltimg, cax = cax, ticks = bounds, orientation='horizontal')


fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
plt.title('Concentration in ' + str(sys.argv[1]) + "/" + str(sys.argv[2])) #El ponemos el titulo a la grafica
plt.xlabel('intensity values of ' + str(sys.argv[2]))#ponemos la etiqueta de eje x
plt.ylabel('intensity values of ' + str(sys.argv[1]))#ponemos la etiqueta de eje y
plt.xticks(np.linspace(0.0,1.0, 10), fontsize =8)
plt.yticks(np.linspace(0.0,1.0, 10), fontsize =8)
ax2.scatter(cv_img2_11, cv_img1_11, s=3)


plt.show()
#Plot de color para la matriz resultante

#pil_cv_img1 = Image.fromarray(cv_img1)
#pil_cv_img2 = Image.fromarray(cv_img2)
#pil_cv_result = Image.fromarray(cv_result)

#pil_cv_img1.show()
#pil_cv_img2.show()
#pil_cv_result.show()