import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from mpl_toolkits.mplot3d import axes3d
import pylab

import pandas as pd
import glob
from PIL import Image
import csv
import numpy as np
import itertools
import re
from numpy.linalg import inv

x = []
y = []
z = []

data = np.genfromtxt('/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/nubes/ground_truth_models/modelo15.obj')
x = data[:,0]
y = data[:,1]
z = data[:,2]


count = 0
with open("/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/nubes/ground_truth_models/model15_scale.xyz", "w") as f:#escala la nube de puntos
    for i in x:
        f.write(str(x[count]/10))
        f.write(str('\t'))
        f.write(str(y[count]/10))
        f.write(str('\t'))
        f.write(str(z[count]/10))
        f.write(str('\t'))
        f.write('\n')
        count = count + 1
f.close()

i=0
#print x[i],y[0],z[0]
#this program scales a point cloud given as input in data = np.genfromtxt...
# the exit point cloud is saved in the directory with open ("...")


