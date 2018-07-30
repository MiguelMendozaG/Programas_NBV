import glob
import re
import os
input_folder = "/home/miguelmg/Documents/CIDETEC/semestre 2/vision 3d/proyecto/6d pose/hinterstoisser/nubes/modelo4/position/pose/*"

files = sorted(glob.glob(input_folder))

for file_ in files:
	number = re.findall(r'\d+',file_)
	#print (number[4])
	if (int(number[4]))%5 == 0:
		print file_
	else:
		os.remove(file_)
