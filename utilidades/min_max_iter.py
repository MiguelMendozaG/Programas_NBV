import numpy as np
import glob
import matplotlib.pyplot as plt

min_ = 50
max_ = 0

def min_max():
	input_folder = "/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/nubes/modelo1"

	folders = sorted(glob.glob(input_folder + "/nbv/*"))
	for folder in folders:
		actual_folder = sorted(glob.glob(folder + '/octo_acum/*'))
		actual_size = len(actual_folder)/3
		#print(len(actual_folder)/3)
		if actual_size < min_:
			min_ = actual_size
			a = folder
		if actual_size > max_:
			max_ = actual_size
			b = folder
		#print('\n')

	print ("min: ", a)
	print("max: ", b)


def min_max_hist(num):
	input_folder = "/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/nubes/modelo" + str(num)

	folders = sorted(glob.glob(input_folder + "/nbv/*"))
	a = np.ones((11))
	overload = 1
	for folder in folders:
		actual_folder = sorted(glob.glob(folder + '/octo_acum/*'))
		actual_size = len(actual_folder)/3
		if actual_size == 2:
			a[actual_size] = a[actual_size] + 1
		elif actual_size == 3:
			a[actual_size] = a[actual_size] + 1
		elif actual_size == 4:
			a[actual_size] = a[actual_size] + 1
		elif actual_size == 5:
			a[actual_size] = a[actual_size] + 1
		elif actual_size == 6:
			a[actual_size] = a[actual_size] + 1
		elif actual_size == 7:
			a[actual_size] = a[actual_size] + 1
		elif actual_size == 8:
			a[actual_size] = a[actual_size] + 1
		elif actual_size == 9:
			a[actual_size] = a[actual_size] + 1
		elif actual_size > 9:
			overload = overload + 1
	a[10] = overload
	if num > 0:
		print(a-1)

def grafica_his():
	np.random.seed(19680801)
	n_bins = 10
	x = np.random.randn(1000, 3)
	fig, axes = plt.subplots(nrows=2, ncols=2)
	ax1 = axes.flatten()
	ax1.hist(x, n_bins, density=True, histtype='bar', stacked=True)
	ax1.set_title('stacked bar')
	
	fig.tight_layout()
	plt.show()

#for i in range(11):
#	min_max_hist(i)

grafica_his()
