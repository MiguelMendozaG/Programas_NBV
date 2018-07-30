import pandas as pd
import glob
from PIL import Image
import csv
import numpy as np
import itertools
import re

images=sorted(glob.glob("/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/train/01/depth/*.png"))

matriz = open("/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/train/01/gt.yml")

"""
data=pd.io.parsers.read_csv(matriz,delimiter="\t")
coords=[[4,5],[6,9]]
for i in coords:
    start_p=int(i[0]);stop_p=int(i[1])
    #df=data[((data.POSITION>=start_p)&(data.POSITION<=stop_p))]
    #df.to_csv(output_table,index=False,sep="\t",header=False,cols=None,mode='a')
"""

def rotacion(num):
    num = num*5+1
    with open("/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/train/01/gt.yml", "r") as text_file:
        for line in itertools.islice(text_file, num, num+1):
            #a = line.strip('- cam_R_m2c: [')
	    a = line.replace('- cam_R_m2c: ','')
            #b = re.findall("\d+\.\d-", a)
	    b = re.findall(r'[+-]?[0-9.]+', a)
	    #b = re.compile(r"[+-]?\d+(?:\.\d+)?", a)
            c = np.matrix([[b[0],b[1],b[2]],[b[3],b[4],b[5]],[b[6],b[7],b[8]]])
            #print c
    text_file.close()
    return c


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
	    with open("/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/nubes/imagen-" + str(count), 'w') as f:
		for i in range (img.width):
		    for j in range (img.height):
		        if px[i,j]:
			    mat = rotacion(count)
			    a_x = (x[i] - cx) * z[i] / fx_d
			    a_y = (y[i] - cy) * z[i] / fy_d
			    a_z = px[i,j]
			    res = mat.I*np.matrix([[a_x],[a_y],[a_z],[1]])
		            f.write(str(i))
		            f.write('\t')
		            f.write(str(j))
		            f.write('\t')
		            f.write(str(float(px[i,j])))
		            f.write('\n')
	    count = count +1
	    f.close()

#a = rotacion(817)
#print a

archivos()

