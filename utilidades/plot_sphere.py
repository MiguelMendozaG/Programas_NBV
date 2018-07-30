import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

points = np.genfromtxt('points_in_sphere.txt')

x = points[:3,0]
y = points[:3,1]
z = points[:3,2]

# Create data
N = 60
g1 = points[0,:]
g2 = points[1,:]
g3 = points[2,:]
g4 = points[3,:]
g5 = points[4,:]
g6 = points[5,:]
g7 = points[6,:]
g8 = points[7,:]
g9 = points[8,:]
g10 = points[9,:]
g11 = points[10,:]
g12 = points[11,:]
g13 = points[12,:]
g14 = points[13,:]
g15 = points[14,:]
g16 = points[15,:]
g17 = points[16,:]
g18 = points[17,:]
g19 = points[18,:]
 
# Create a sphere
r = 0.35
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
x_sphere = r*sin(phi)*cos(theta)
y_sphere = r*sin(phi)*sin(theta)
z_sphere = r*cos(phi)
 
data = (g1, g2, g3, g4, g5, g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,g16,g17,g18,g19)
colors = ("red", "green", "blue", "cyan", "yellow", "k", "magenta", "white","violet","pink","gray","peru","lightblue","lime","indigo")
groups = ("0", "1", "2", "3", "4", "5","6", "7", "8", "9", "10", "11","12", "13", "14", "15","16", "17", "18", "19") 
 
# Create plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
ax = fig.gca(projection='3d')
 
for data, color, group in zip(data, colors, groups):
    x, y, z = data
    ax.scatter(x, y, z, alpha=0.8, c=color, edgecolors='none', s=30, label=group)
    
#ax.plot_surface(x_sphere, y_sphere, z_sphere,  rstride=1, cstride=1, color='c', alpha=0.6,linewidth=0)

plt.plot([0],[0],[0])
plt.title('Matplot 3d scatter plot')
plt.legend(loc=2)
plt.show()
