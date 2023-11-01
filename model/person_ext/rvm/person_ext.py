#! /usr/bin/python3
# coding=utf-8

import os
import cv2
import torch
from .model import MattingNetwork
from .inference import convert_video


def person_ext_rvm(vid, input_path, person_folder, frame_size_threshold=800):
	print(f"\t Start silhouette extraction.")

	silhouette_path = os.path.sep.join([person_folder, 'silhouette', vid])
	image_path = os.path.sep.join([person_folder, 'image', vid])
	os.makedirs(silhouette_path, exist_ok=True)
	os.makedirs(image_path, exist_ok=True)

	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# load mobilenetv3 model
	model = MattingNetwork('mobilenetv3').eval().to(device)
	model_path = os.path.sep.join([os.path.dirname(__file__), "work", "checkpoint", "rvm_mobilenetv3.pth"])
	model.load_state_dict(torch.load(model_path))

	# calc video input_resize
	cap = cv2.VideoCapture(input_path)
	frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	cap.release()
	if max(frame_width, frame_height) <= frame_size_threshold:
		input_resize = None
	else:
		if frame_width > frame_height:
			ratio = frame_width / frame_size_threshold
		else:
			ratio = frame_height / frame_size_threshold
		input_resize = (int(frame_width // ratio), int(frame_height // ratio))
	print(f"\t {input_resize=}")

	# output png_sequence
	convert_video(
		model,  #
		input_source=input_path,  
		# num_workers=1, 
		input_resize=input_resize,  
		output_type='png_sequence', 
		output_background='default', 
		output_composition=image_path, 
		output_alpha=silhouette_path,  
		downsample_ratio=None,  
		seq_chunk=4,  
		progress=True 
	)


# if __name__ == '__main__':
# 	input_path = "data\\upload\\14\\video\\b2092bb2.avi"
# 	person_folder = "data\\upload\\14"
# 	vid = "b2092bb2"
# 	person_ext_rvm(vid, input_path, person_folder)
