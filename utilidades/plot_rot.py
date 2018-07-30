import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from mpl_toolkits.mplot3d import axes3d
import pylab

x2 = []
y2 = []
z2 = []
x = []
y = []
z = []

data = np.genfromtxt('/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/nubes/modelo6/absolute/model/imagen-0.xyz')
x = data[:,0]
y = data[:,1]
z = data[:,2]


data2 = np.genfromtxt('/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/nubes/modelo6/absolute/model/imagen-5.xyz')
x2 = data2[:,0]
y2 = data2[:,1]
z2 = data2[:,2]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.plot_wireframe(x,y,z, rstride=10, cstride=10)
ax.scatter(x2,y2,z2)
ax.scatter(x,y,z)
#ax.plot_wireframe(x2,y2,z2, rstride=10, cstride=10)

plt.show()
