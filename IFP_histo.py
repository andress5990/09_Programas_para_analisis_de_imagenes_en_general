import numpy
from PIL import Image
import matplotlib.pyplot as plt
import itertools
import matplotlib.mlab as mlab
from scipy.stats import norm
import pdb

def luminosity( l, k):
    r, g, b=pixel_values[l][k]
    lum=0.299*r +0.587*g+0.114*b
    return lum


L=[] #Matrix
im=Image.open('Color.jpg')  #Leyendo la imagen como una lista de listas con entradas internas de 4 entradas.

#im=Image.open('starry_night.jpg')
im_n = numpy.array(im)  #Convirtiendo el array como una matriz
width, height=im.size   #Caracterizando el tamano de la matriz
#r, g, b=im.split()	#Alternativa para la linea inferior
pixel_values = list(im.getdata())   #Obteniendo las cuatro entradas de la matriz im
#La linea anterior da una matriz con los 3 canales mezcaldos
pixel_values2 = numpy.array(pixel_values).reshape((width, height, 3))  #Reescribiendo las variables a r, g, b sin alfa.
#La linea anterior separa los 3 canales  
r, g, b=pixel_values2[0][2]
pdb.set_trace()

L=[[int(luminosity(i,j)) for j in range(height)] for i in range(width) ]   #Calculando la luminosidad y dejando solo 3 decimales por cuestion de orden



print ('Width, Height: ', im.size)  #Verificando tamano de la matriz


print (numpy.matrix(L))  #Formato para comprobar la matriz
#print (im_n.shape, im_n.dtype)
#numpy.savetxt('/Matrix_reposi/test', L, fmt='%.2f', delimiter=' ', header='prueba.jpg') # Guardando la matriz L como un txt fmt es el formato y se le puso un titulo

holder=[]
#cm = plt.cm.get_cmap('')

for i, j in itertools.product(range(width), range(height)):
	holder.append(L[i][j])
#V_a=np.arange(-255.0,255.0, step=1.0)


n, bins, patches = plt.hist(holder, bins= 60, normed=1, alpha=0.75)
# best fit of data
(mu, sigma) = norm.fit(holder)

# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)

plt.title(r'$\mathrm{Histogram\ of\ Luminosity:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))
plt.xlabel("Luminosity")
plt.savefig('Cut_6.jpg')
plt.show()

