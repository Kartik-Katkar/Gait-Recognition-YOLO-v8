import os
from PIL import Image
import numpy as np


def cut_image(path, cut_path, size):
	for (root, dirs, files) in os.walk(path):
		temp = root.replace(path, cut_path)
		if not os.path.exists(temp):
			os.makedirs(temp)
		for file in files:
			image, flag = cut(Image.open(os.path.join(root, file)))
			if not flag:
				Image.fromarray(image).convert('L').resize((size, size)).save(os.path.join(temp, file))

	pass


def cut(image):

	image = np.array(image)


	# print(image)
	# print(image.sum(axis=1))
	# print(image.sum(axis=1)!=0)
	height_min = (image.sum(axis=1) != 0).argmax()  
	height_max = ((image.sum(axis=1) != 0).cumsum()).argmax()  
	width_min = (image.sum(axis=0) != 0).argmax()
	width_max = ((image.sum(axis=0) != 0).cumsum()).argmax()
	head_top = image[height_min, :].argmax()
	
	size = height_max - height_min
	temp = np.zeros((size, size))

	
	l1 = head_top - width_min
	r1 = width_max - head_top
	
	flag = False
	if size <= width_max - width_min or size // 2 < r1 or size // 2 < l1:
		flag = True
		return temp, flag
	# centroid = np.array([(width_max+width_min)/2,(height_max+height_min)/2], dtype='int')
	temp[:, (size // 2 - l1):(size // 2 + r1)] = image[height_min:height_max, width_min:width_max]

	return temp, flag


def gei(cut_path, data_path, size):
	
	for (root, dirs, files) in os.walk(cut_path):
		temp = root.replace(cut_path, data_path)
		if not os.path.exists(temp):
			os.makedirs(temp)
		GEI = np.zeros([size, size])
		if len(files) != 0:
			for file in files:
				GEI += Image.open(os.path.join(root, file)).convert('L')
			GEI /= len(files)
			Image.fromarray(GEI).convert('L').resize((size, size)).save(os.path.join(temp, 'gei.png'))
	pass


if __name__ == '__main__':
	INPUT_PATH = r"D:\Development\Python\Pycharm\AI\09GaitRecognitionSystem\GaitRecognitionSystem\data\upload\124\silhouette"
	OUTPUT_PATH = r"D:\Development\Python\Pycharm\AI\09GaitRecognitionSystem\GaitRecognitionSystem\data\predata"
	# OUTPUT_PATH2 = r"D:\Development\Python\Pycharm\AI\09GaitRecognitionSystem\GaitRecognitionSystem\data"
	cut_image(INPUT_PATH, OUTPUT_PATH, 128)
	# gei(OUTPUT_PATH, OUTPUT_PATH2, 128)
