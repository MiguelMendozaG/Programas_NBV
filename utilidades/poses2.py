import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from mpl_toolkits.mplot3d import axes3d
import pylab
from PIL import Image


x=[]
y=[]
z=[]
x_plot = []
y_plot = []
z_plot = []
x_plot2 = []
y_plot2 = []
z_plot2 = []
fx_d = 572.41140000
fy_d = 573.57043000
cx = 325.26110000
cy = 242.04899000

mat_int = np.matrix([[572.41140000, 0.00000000, 325.26110000],[0.00000000, 573.57043000, 242.04899000],[0.00000000, 0.00000000, 1.00000000]])

mat = np.matrix([[0.93682336, -0.34980278, 0.00000000,0],[-0.30293596, -0.81130712, -0.50001056,0],[ 0.17490508, 0.46842157, -0.86601931,400],[0,0,0,1]])  #img202

mat2 = np.matrix([[0.00000000, 1.00000000, 0.00000000,0],[0.77289053, 0.00000000, -0.63453938,0],[-0.63453938, 0.00000000, -0.77289053,400],[0,0,0,1]]) #img269

mat_I = np.matrix([[1,0,0],[0,1,0],[0,0,1]])

tras = np.matrix([[0],[0],[400]])

#c_tilde = -(mat.I) * tras


data = np.genfromtxt("/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/nubes/raw/imagen-202")
x = data[:,0]
y = data[:,1]
z = data[:,2]

#print len(x)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

with open("/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/nubes/rotacion/imagen-" + str(202), 'w') as f:
    for i in range(len(x)):
        
        a_x = (x[i] - cx) * z[i] / fx_d
        a_y = (y[i] - cy) * z[i] / fy_d
        a_z = z[i]
        res = mat.I*np.matrix([[a_x],[a_y],[a_z],[1]])
        """
        x_tilde = mat.I * np.matrix([[x[i]],[y[i]],[z[i]]])
	mue_n = -c_tilde[2] / x_tilde[2]
	res = np.matrix([[mue_n,0,0],[0,mue_n,0],[0,0,mue_n]]) * (mat.I) * np.matrix([[x[i]],[y[i]],[z[i]]]) + c_tilde
	"""
        f.write(str((res[0,0])))
        f.write('\t')
        f.write(str((res[1,0])))
        f.write('\t')
        f.write(str((res[2,0])))
        f.write('\n')
        x_plot.append((res[0,0]))
        y_plot.append((res[1,0]))
        z_plot.append(res[2,0])

f.close()
# Plot a basic wireframe.
ax.plot_wireframe(x_plot,y_plot,z_plot, rstride=10, cstride=10)
#ax.scatter(x_plot, y_plot, z_plot, 'r', 'o')



data = np.genfromtxt("/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/nubes/raw/imagen-269")
x = data[:,0]
y = data[:,1]
z = data[:,2]

print len(x)

with open("/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/nubes/rotacion/imagen-" + str(269), 'w') as f:
    for i in range(len(x)):
	
        a_x = (x[i] - cx) * z[i] / fx_d
        a_y = (y[i] - cy) * z[i] / fy_d
        a_z = z[i]
        res = mat2.I *np.matrix([[a_x],[a_y],[a_z],[1]])
	"""
	x_tilde = mat.I * np.matrix([[x[i]],[y[i]],[z[i]]])
	mue_n = -c_tilde[2] / x_tilde[2]
	res = np.matrix([[mue_n,0,0],[0,mue_n,0],[0,0,mue_n]]) * (mat.I) * np.matrix([[x[i]],[y[i]],[z[i]]]) + c_tilde
	"""
        f.write(str((res[0,0])))
        f.write('\t')
        f.write(str((res[1,0])))
        f.write('\t')
        f.write(str((res[2,0])))
        f.write('\n')
	
        x_plot2.append((res[0,0]))
        y_plot2.append((res[1,0]))
        z_plot2.append(res[2,0])
	


f.close()

#plt.plot(x_plot, y_plot, z_plot, "o")
print len(x_plot)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# Plot a basic wireframe.
ax.plot_wireframe(x_plot2,y_plot2,z_plot2, rstride=10, cstride=10)
#ax.scatter(x_plot, y_plot, z_plot, 'r', 'o')
plt.show()

