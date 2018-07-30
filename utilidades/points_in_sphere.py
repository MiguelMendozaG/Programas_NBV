import math, random
#Creates n points distributed on the surface of a sphere of radio r #
#https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere#

from numpy import pi, cos, sin, arccos, arange
import mpl_toolkits.mplot3d
import matplotlib.pyplot as pp
import numpy as np

num_pts = 20
indices = arange(0, num_pts, dtype=float) + 0.5
r = 0.4

phi = arccos(1 - 2*indices/num_pts)
theta = pi * (1 + 5**0.5) * indices

x, y, z = (cos(theta) * sin(phi))*r, (sin(theta) * sin(phi))*r, (cos(phi))*r;

coords = []
for i in range(num_pts):
	#print (x[i])
	coords.append((x[i],y[i],z[i]))

#print (coords)

#np.savetxt('points_in_sphere.txt', coords, fmt='%f')

print (math.sqrt(x[0]**2 + y[0]**2 + z[0]**2))

pp.figure().add_subplot(111, projection='3d').scatter(x, y, z);
pp.show()

