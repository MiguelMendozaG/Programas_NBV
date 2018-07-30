import numpy as np
import glob
from shutil import copyfile #library to copy files
from scipy.spatial import distance
import os
import mpl_toolkits.mplot3d
import matplotlib.pyplot as pp

#This programm labels the positions for every pose in the octree files

folder = sorted(glob.glob('/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/nubes/modelo2/nbv/*'))
output_folder = "/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/nubes/dataset/2/"
numpy_file = "/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/nubes/dataset/"
output_dataset = "/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/nubes/dataset/dataset_model2"

reference_points = np.genfromtxt('points_in_sphere.txt')
x = reference_points[:,0]
y = reference_points[:,1]
z = reference_points[:,2]

print (len(reference_points))
i=0

stop = 0
stop2 = 0
pos_indice = 0
folder_indice = 0
dataset = []
for folders in folder:
	#if not os.path.exists(output_folder + str(folder_indice)):
		#os.makedirs(output_folder + str(folder_indice))
	#if stop < 1:
	sub_folder = sorted(glob.glob(folders + '/octo_acum/*'))
	num_files =	int(len(sub_folder)/3)
	for file_ in range(num_files):
		file_ = file_ + 1
		octree_file = folders + '/octo_acum/octomap_acum_' + str(file_) + '.txt'
		pos_file = np.genfromtxt(folders + "/poses/pose_orientation/pose_orn" + str(file_) + ".dat")
		tuple_first = np.loadtxt(octree_file)
		pos_coord = pos_file[0]	
		num_point = 0
		min_distance = 1
		for points in reference_points:
			#print points , pos_coord
			distance_ = distance.euclidean(points, pos_coord)
			#print distance_
			if (distance_ < min_distance):
				min_distance = distance_
				pos_indice = num_point
			num_point = num_point + 1
		#print pos_coord
		clase = np.zeros((1,20))
		clase[0,pos_indice] = 1
		#print clase
		dst = output_folder + str(pos_indice) + "_" + str(folder_indice) + "_"  + str(file_) + ".txt"
		#print octree_file
		#print dst
		copyfile(octree_file, dst)
		dataset.append([tuple_first, clase])
		stop = stop + 1
	
	folder_indice = folder_indice + 1

np.save(output_dataset, dataset)

"""
fig = pp.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[0:13], y[0:13], z[0:13], c = 'r')
ax.scatter(x[13], y[13], z[13], c = "g")
ax.scatter(x[14:20], y[14:20], z[14:20], c= 'k')
ax.scatter(0.241617,  -0.317545,   0.0280363)
ax.set_xlim([-0.5,0.5])
ax.set_ylim([-0.5,0.5])
ax.set_zlim([-0.5,0.5])
pp.show()
"""

print ("Finish")
#print (dataset[0][1])








