import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

#a = np.array([[1.23,2.21,3.29,4.45],[2,3,4,5],[3,4,5,6]])
#np.savetxt('test_array.txt', a, fmt='%f')

b = np.loadtxt('/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/nubes/dataset/test_DNN/octree_pruned15.txt', dtype = float)
#b = np.loadtxt('/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/nubes/dataset/1/0_251_1.txt', dtype = float)
#####inputs#########
valx = 0.086408
min_octrees_file = -valx
max_octrees_file = valx
min_cubo = 0
max_cubo = 32

m = (max_cubo - min_cubo) / (max_octrees_file - min_octrees_file)

x = b[:,0]
y = b[:,1]
z = b[:,2]
v = b[:,3]

x_new = []
y_new = []
z_new = []
x_new2 = []
y_new2 = []
z_new2 = []
v_new = []
output_cube = np.zeros((32,32,32))
large = len(v)
print ("large= ", large)

for i in range(large):
    x_cube = int((x[i]*m*2+32)/2)
    y_cube = int((y[i]*m*2+32)/2)
    z_cube = int((z[i]*m*2+32)/2)
    x_new.append((x[i]*m*2+32)/2)
    y_new.append((y[i]*m*2+32)/2)
    z_new.append((z[i]*m*2+32)/2)
    output_cube[x_cube][y_cube][z_cube] = v[i]

print (output_cube.shape)
print (output_cube[31][31][31])

np.save('octomap1', output_cube)

cube = np.load('octomap1.npy')
cube_reshape = cube.reshape(32*32*32,1)
np.savetxt('octomap1_reshape.txt', cube_reshape ,fmt='%f')


a=1
b=1
c=1
for i in range(large):
    if v[i] < 0.5:
        a=1 + a
        #x_new2.append(x_new[i])
        #y_new2.append(y_new[i])
        #z_new2.append(z_new[i])
        #v_new.append(0)
    elif v[i] >= 0.5 and v[i] < 0.6:
        b=1 + b
        x_new2.append(x_new[i])
        y_new2.append(y_new[i])
        z_new2.append(z_new[i])
        v_new.append(0)
    elif v[i] >= 0.6:
        c=1+c
        x_new2.append(x_new[i])
        y_new2.append(y_new[i])
        z_new2.append(z_new[i])
        v_new.append(255)

print ("a= ",a, "b= ",b, "c= ",c, "total= ", a+b+c-3)

#plt.imshow(output_cube[:][:][9])
#plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_new2,y_new2,z_new2, c = v_new)
plt.show()

