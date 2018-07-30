import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import glob

input_folder = "/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/nubes/modelo2"

output_folder = "/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/nubes/output_dataset"

total_inputs = 1312
folders = sorted(glob.glob(input_folder + "/nbv/*"))
print (folders[0])

min_octrees_file = -0.106268
max_octrees_file = 0.106268
min_cubo = 1
max_cubo = 32

m = (max_cubo - min_cubo) / (max_octrees_file - min_octrees_file)

for folder in folders:
	info_folder = sorted(glob.glob(folder + '/octo_acum/*'))
    info_folder_size = len(info_folder)/3
		for info in range(info_folder_size):
			x_new = []
			y_new = []
			z_new = []
			x_new2 = []
			y_new2 = []
			z_new2 = []
			v_new = []
			output_cube = np.zeros((32,32,32))
			large = len(v)
			actual_file = folder + '/octo_acum/octomap_acum_' + str(info) + '.txt'
			actual_pose = folder + '/poses/pose_orientation/pose_orn' + str(info) + '.dat'
			b = np.loadtxt(actual_file, dtype = float)
			x = b[:,0]
			y = b[:,1]
			z = b[:,2]
			v = b[:,3]
			for i in range(large):
				x_cube = int((x[i]*m*2+31)/2)
				y_cube = int((y[i]*m*2+31)/2)
				z_cube = int((z[i]*m*2+31)/2)
				x_new.append((x[i]*m*2+31)/2)
				y_new.append((y[i]*m*2+31)/2)
				z_new.append((z[i]*m*2+31)/2)
				output_cube[x_cube][y_cube][z_cube] = v[i]

print (output_cube.shape)
print (output_cube[31][31][31])

np.save('octomap1', output_cube)

#cube = np.load('octomap1.npy')
#cube_reshape = cube.reshape(32*32*32,1)
#np.savetxt('octomap1_reshape.txt', cube_reshape ,fmt='%f')

