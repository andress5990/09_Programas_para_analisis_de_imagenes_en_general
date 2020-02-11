#Andres Chavarria Sibaja-----------------------------------------------------------------------------------------
#Programa para mostrar la concentracion de sustancia en imagenes a partir de la intensidad-----------------------
#2019------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------
#Libraries used--------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------

import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

#----------------------------------------------------------------------------------------------------------------
#Function Definitions--------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------

def cal_luminosity(file_name):
    
    cv_img = cv2.imread(file_name)#np.array con shape(filas, columnas, canales)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) #volvemos el valor del np.array a RGB para procesar
    cv_img = cv_img[40:407, 0:500]#Esta linea corta la imagen y elimina la franja blanca de arriba
    #b, g, r = cv2.split(cv_img)#separamos los canales en gbr
    b = cv_img[:,:,0]#Estas linea separa el canal b de la imagen 
    g = cv_img[:,:,1]#Esta linea separa el canal g de la imagen
    r = cv_img[:,:,2]#Esta linea separa el canal r de la imagen 
    
    holder = np.ones(cv_img[:,:,0].shape)#Creamos una matriz de 0s con el mismo tamano de la matriz de imagen
                                         #para que funcione como holder
    #Recorremos la imagen por den fila-columna
    for i in range(b.shape[0]):#Filas
        for j in range(b.shape[1]):#Columnas
            #Calculamos la fila con alguna formula
            lum=0.2126*r[i][j] + 0.7152*g[i][j] + 0.0722*b[i][j] #Photometric/digital ITU BT.709
            #lum=0.299*r[i][j] +0.587*g[i][j]+0.114*b[i][j] #Digital ITU BT.601 
                                                            #(gives more weight to the R and B components)
            holder[i][j] = lum#cambiamos la entrada del la matriz holder con el valor calculado
    
    #Normalizamos la matriz
    maxholder = np.amax(holder)#Sacamos el valor maximo existente en el array para usarlo en la normalizacion
    #Recorremos la imagen por den fila-columna
    for i in range(b.shape[0]):#Filas
        for j in range(b.shape[1]):#Columnas
            holder[i][j] = holder[i][j]/maxholder#Dividimos el valor de la matriz en la coordenada i,j
                                                 #el max holder
   
    minholder = np.amin(holder)#Calculamos el minimo de la matriz normalizada
    maxholder = np.amax(holder)#Calculamos el maximo de la matriz normalizada    

    return(holder, maxholder, minholder)#Retornamos la matriz holder 

#----------------------------------------------------------------------------------------------------------------
#Main execution program------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------

image_name = sys.argv[1]#Esto lee cual es el nombre de la imagen a analizar

#creamos ambas matrices a partir de del calculo de luminosidad (grayscale)
cv_img = cal_luminosity(image_name)[0]#Esta es la imagen en blanco y negro normalizada que devuelve cal_luminosity
cv_imgMax = cal_luminosity(image_name)[1]#este es el valor del maximo que devuelve cal_luminosity
cv_imgMin = cal_luminosity(image_name)[2]#este es el valor del minimo que devuelve cal_luminosity

bounds = np.linspace(cv_imgMin, cv_imgMax, 20, endpoint=True)#Creamos un array de valores para la barra 
                                                             #de colores
bounds = np.around(bounds, 1)#redondeamos los valores de bound para que sea mas presentable

yscale = len(cv_img[:,1])#Tomamos el largo de las filas de la imagen
xscale = len(cv_img[1,:])#Tomamos el largo de las columnas de la imagen
                         #xscale hay que ponerlo asi porque si no, rota la imagen

ax1 = plt.subplot(111) #Creamos el espacio para plottear
ax1.set_xlim(0, xscale)#definimos el limite de x para el eje x
ax1.set_ylim(yscale,0)#definimos el limite de y para el eje y
plt.title('Concentration in ' + str(sys.argv[1])) #El ponemos el titulo a la grafica
plt.ylabel('pixels y coordinate')#ponemos la etiqueta de eje y
plt.xlabel('pixels x coordinate')#ponemos la etiqueta de eje x
pltimg = ax1.imshow(cv_img, cmap = plt.cm.jet)#graficamos la imagen
divider = make_axes_locatable(ax1)#Hacemos que las etiquetas que se definen debajo de ca, 
                                  #sean ubicables donde se quiera
cax = divider.append_axes("right", size="5%", pad=0.1)#Definimos el tamaño posicion 
                                                      #y grosor de la barra de color
plt.colorbar(pltimg, cax = cax, ticks = bounds)#Creamos la barra de color
#plt.savefig()
plt.show()#Mostramos todo el grafico completo
