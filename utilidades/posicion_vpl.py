import pandas as pd
import glob
from PIL import Image
import csv
import numpy as np
import itertools
import re
from numpy.linalg import inv
import os

x = np.matrix([[0],[0],[-400],[1]])
trans = np.matrix([[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

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
            c = np.matrix([[float(b[0]),float(b[1]),float(b[2]),0],[float(b[3]),float(b[4]),float(b[5]),0],[float(b[6]),float(b[7]),float(b[8]),0],[0,0,0,1]]) #se guarda la matriz de transformacion en c
	    c = c
            #print c
    text_file.close()
    return c

def archivos2(): #guarda todas las coordenadas en un solo archivo (imagen_origenes.dat)
    count = 0
    with open("/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/nubes/posicion/pose/imagen_origines_2.dat", 'w') as f:
        for image in images:
            mat = rotacion(count)
            res =mat.I * x 
            f.write(str(res[0,0]/1000))
            f.write('\t')
            f.write(str(res[1,0]/1000))
            f.write('\t')
            f.write(str(res[2,0]/1000))
            f.write('\t')
            f.write('\n')
            count = count + 1
        f.close()
        


def archivos():#guarda un archivo por imagen, cada rchivo contiene posicion x, y, z
    count = 0
    for image in images:
        
	#print "imagen" ,count
	with open("/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/nubes/posicion/pose/imagen_origin_" + str(count) + str('.dat'), 'w') as f:
        #with open(direc, 'w') as f:
            mat = rotacion(count)
            res =mat.I * x 
            f.write(str(res[0,0]/1000))
            f.write('\t')
            f.write(str(res[1,0]/1000))
            f.write('\t')
            f.write(str(res[2,0]/1000))
            f.write('\t')
            f.write('\n')
	    f.close()
        count = count + 1

archivos2()
a = trans * rotacion(100)
print a
#archivos()
