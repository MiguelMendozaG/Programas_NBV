import math
from math import atan2
import pandas as pd
import glob
from PIL import Image
import csv
import numpy as np
import itertools
import re
from numpy.linalg import inv


images=sorted(glob.glob("/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/train/01/depth/*.png"))

def rotacion(num):
    num = num*5+1
    with open("/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/train/01/gt.yml", "r") as text_file:
        for line in itertools.islice(text_file, num, num+1):
            #a = line.strip('- cam_R_m2c: [')
	    a = line.replace('- cam_R_m2c: ','') #reemplaza la cadena '- cam_R_m2c: ' por un espacio ' ' 
            #b = re.findall("\d+\.\d-", a)
	    b = re.findall(r'[+-]?[0-9.]+', a) #encuentra todos los numeros flotantes positivos y negativos y los guarda en una matriz
	    #b = re.compile(r"[+-]?\d+(?:\.\d+)?", a)
            c = np.matrix([[float(b[0]),float(b[1]),float(b[2])],[float(b[3]),float(b[4]),float(b[5])],[float(b[6]),float(b[7]),float(b[8])]]) #se guarda la matriz de transformacion en c
	    c = c
            #print c
    text_file.close()
    return c

def archivos2():
    fx_d = 572.41140000
    fy_d = 573.57043000
    cx = 325.26110000
    cy = 242.04899000
    count = 0
    with open("/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/nubes/posicion/orientaciones/imagen-orientaciones.dat", 'w') as f:
	for image in images:
	    print "imagen" ,count
	    img = Image.open(image)
	    px = img.load()
            mat = rotacion(count)
            yaw = atan2(mat[1,0],mat[0,0])
            pitch = atan2(-mat[2,0],pow(pow(mat[2,1],2)+pow(mat[2,2],2),0.5))
            roll = atan2(mat[2,1],mat[2,2])
	    #res = mat.I*np.matrix([[a_x],[a_y],[a_z],[1]])
	    f.write(str((yaw)))
	    f.write('\t')
	    f.write(str((pitch)))
	    f.write('\t')
	    f.write(str((roll)))
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
	    with open("/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/nubes/posicion/orientaciones/imagen-ori-" + str(count) + str('.dat'), 'w') as f:
                mat = rotacion(count)
                yaw = atan2(mat[1,0],mat[0,0])
                pitch = atan2(-mat[2,0],pow(pow(mat[2,1],2)+pow(mat[2,2],2),0.5))
                roll = atan2(mat[2,1],mat[2,2])
		#res = mat.I*np.matrix([[a_x],[a_y],[a_z],[1]])
		f.write(str(math.degrees(yaw)))
		f.write('\t')
		f.write(str(math.degrees(pitch)))
		f.write('\t')
		f.write(str(math.degrees(roll)))
		f.write('\n')
                f.close()
	    count = count +1

archivos2()

math.pi
a = atan2(0,1)
print math.degrees(a)
