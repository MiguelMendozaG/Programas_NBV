import pandas as pd
import glob
from PIL import Image
import csv
import numpy as np
import itertools
import re
from numpy.linalg import inv

images=sorted(glob.glob("/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/train/01/depth/*.png"))

matriz = open("/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/train/01/gt.yml")

x_min = 200
y_min = 130
x_max = 450
y_max = 380


#mat = np.matrix([[0.93682336, -0.34980278, 0.00000000,0],[-0.30293596, -0.81130712, -0.50001056,0],[ 0.17490508, 0.46842157, -0.86601931,400],[0,0,0,1]])  #img202

def rotacion(num):
    num = num*5+1
    with open("/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/train/01/gt.yml", "r") as text_file:
        for line in itertools.islice(text_file, num, num+1):
            #a = line.strip('- cam_R_m2c: [')
	    a = line.replace('- cam_R_m2c: ','') #reemplaza la cadena '- cam_R_m2c: ' por un espacio ' ' 
            #b = re.findall("\d+\.\d-", a)
	    b = re.findall(r'[+-]?[0-9.]+', a) #encuentra todos los numeros flotantes positivos y negativos y los guarda en una matriz
	    #b = re.compile(r"[+-]?\d+(?:\.\d+)?", a)
            c = np.matrix([[float(b[0]),float(b[1]),float(b[2]),0],[float(b[3]),float(b[4]),float(b[5]),0],[float(b[6]),float(b[7]),float(b[8]),400],[0,0,0,1]]) #se guarda la matriz de transformacion en c
	    c = c
            #print c
    text_file.close()
    return c

def archivos5():#crea nube de puntos por cada imagen, guarda solamente los puntos correspondientes al modelo (sin plano de fondo)
        fx_d = 572.41140000
	fy_d = 573.57043000
	cx = 325.26110000
	cy = 242.04899000
	count = 0
	for image in images:
	    print "imagen" ,count
	    img = Image.open(image)
	    px = img.load()
	    with open("/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/nubes/absolutas/modelo/imagen-" + str(count) + str('.dat'), 'w') as f:
		for i in range (img.width):
		    for j in range (img.height):
                        if px[i,j]:
  			    mat = rotacion(count)
			    a_x = (i - cx) * px[i,j] / fx_d
			    a_y = (j - cy) * px[i,j] / fy_d
			    a_z = px[i,j]
			    res = mat.I*np.matrix([[a_x],[a_y],[a_z],[1]])
		            f.write(str(res[0,0]/1000))
		            f.write('\t')
                            f.write(str(res[1,0]/1000))
		            f.write('\t')
		            f.write(str(res[2,0]/1000))
		            f.write('\n')
	    count = count +1
	    f.close()

def archivos4():#crea nube de puntos por cada imagen, (recortada). Solamente genera los planos de fondo
        fx_d = 572.41140000
	fy_d = 573.57043000
	cx = 325.26110000
	cy = 242.04899000
	count = 0
	for image in images:
	    print "imagen" ,count
	    img = Image.open(image)
	    px = img.load()
	    with open("/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/nubes/absolutas/planos/imagen-" + str(count) + str('.dat'), 'w') as f:
		for i in range (img.width):
		    for j in range (img.height):
                        if (((i>=x_min) and (i<=x_max)) and ((j>=y_min) and (j<=y_max))):
                            if px[i,j] == 0:
                                px[i,j] = 3000
  			        mat = rotacion(count)
			        a_x = (i - cx) * px[i,j] / fx_d
			        a_y = (j - cy) * px[i,j] / fy_d
			        a_z = px[i,j]
			        res = mat.I*np.matrix([[a_x],[a_y],[a_z],[1]])
		                f.write(str(res[0,0]/1000))
		                f.write('\t')
                                f.write(str(res[1,0]/1000))
		                f.write('\t')
		                f.write(str(res[2,0]/1000))
		                f.write('\n')
	    count = count +1
	    f.close()


def archivos3():#crea nube de puntos por cada imagen, y recorta la imagen
        fx_d = 572.41140000
	fy_d = 573.57043000
	cx = 325.26110000
	cy = 242.04899000
	count = 0
	for image in images:
	    print "imagen" ,count
	    img = Image.open(image)
	    px = img.load()
	    with open("/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/nubes/absolutas/cortadas/imagen-" + str(count) + str('.dat'), 'w') as f:
		for i in range (img.width):
		    for j in range (img.height):
                        if (((i>=x_min) and (i<=x_max)) and ((j>=y_min) and (j<=y_max))):
                            if px[i,j] == 0:
                                px[i,j] = 3000
  			    mat = rotacion(count)
			    a_x = (i - cx) * px[i,j] / fx_d
			    a_y = (j - cy) * px[i,j] / fy_d
			    a_z = px[i,j]
			    res = mat.I*np.matrix([[a_x],[a_y],[a_z],[1]])
		            f.write(str(res[0,0]/1000))
		            f.write('\t')
                            f.write(str(res[1,0]/1000))
		            f.write('\t')
		            f.write(str(res[2,0]/1000))
		            f.write('\n')
	    count = count +1
	    f.close()


def archivos2(): #crea un archivo con la nube de puntos de todo el modelo
        fx_d = 572.41140000
	fy_d = 573.57043000
	cx = 325.26110000
	cy = 242.04899000
	count = 0
	with open("/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/nubes/absolutas/imagen-nube_de_puntos.dat", 'w') as f:
	    for image in images:
	        print "imagen" ,count
	        img = Image.open(image)
	        px = img.load()
	        for i in range (img.width):
		    for j in range (img.height):
                        if px[i,j]:
		            mat = rotacion(count)
                            a_x = (i - cx) * px[i,j] / fx_d
			    a_y = (j - cy) * px[i,j] / fy_d
			    a_z = px[i,j]
			    res = mat.I*np.matrix([[a_x],[a_y],[a_z],[1]])
		            f.write(str(res[0,0]/1000))
		            f.write('\t')
                            f.write(str(res[1,0]/1000))
		            f.write('\t')
		            f.write(str(res[2,0]/1000))
		            f.write('\n')
                count = count +1
	f.close()


def archivos():
        fx_d = 572.41140000
	fy_d = 573.57043000
	cx = 325.26110000
	cy = 242.04899000
	count = 0
	for image in images:
	    print "imagen" ,count
	    img = Image.open(image)
	    px = img.load()
	    with open("/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/nubes/absolutas/plano_con_modelo/imagen-" + str(count) + str('.dat'), 'w') as f:
		for i in range (img.width):
		    for j in range (img.height):
                        if px[i,j] == 0:
                            px[i,j] = 3000
		        #if px[i,j]:
			mat = rotacion(count)
			a_x = (i - cx) * px[i,j] / fx_d
			a_y = (j - cy) * px[i,j] / fy_d
			a_z = px[i,j]
			res = mat.I*np.matrix([[a_x],[a_y],[a_z],[1]])
		        f.write(str(res[0,0]/1000))
		        f.write('\t')
                        f.write(str(res[1,0]/1000))
		        f.write('\t')
		        f.write(str(res[2,0]/1000))
		        f.write('\n')

	    count = count +1
	    f.close()
    

#a = rotacion(817)
#print a

archivos2()

