
from util import *


random_seed = 7 # seed for stabilized randomize

#actions can be combination of some action like patching or striding

#function names and requirements are explained below
"""
action type : patch
		-- needs patch size => size of one crop
		-- needs function args => stepsize, if stepsize is less than patch size, it will overlap
		-- optional validation size => if validation 0.0, it wont split data and create folder like output/labels
									   if validation is >0 it will split data randomly and create folders like output/train/labels and output/validation/labels


action type : augmentaion
		-- no need patch size
		-- no need args
		-- validation must be 0.0, augmentation applied only Train set

action type : kmeans
		-- no need patch size
		-- no need args => centers are initialized on utils file
		-- validation rules are same as patching

action type : nuclei_detection
		-- output folder name for nuclei centers
		-- no need patch size
		-- 1st arg is for trained model name placed on /models/ folder
		   2nd arg is for output images if requested ( takes much more time than detecting centers )
		   3rd arg is for output image folder name
		-- validation must be 0.0

action type : image_to_nucleus
		-- output folder name for patched nuclei images
		-- needs patch size => size of one crop
		-- needs function args => 1st found nuclei centers with data structure like data/labels
								  2nd to be used rate of nucleus centers. 1.0 mean all centers, if need big patches, this rate must be reduced
		-- validation rules are same as patching

"""


#(input_folder_name, 	output_folder_name, 			function_name, 		patch_sizes, function_args, validation_size)
actions = [	
("Train_preproc", 		"patched_files", 				"patch", 			(256,256),   (256,256), 	0.2),
("Train_preproc", 		"patched_files_with_stride",    "patch", 			(1024,1024), (256,256),  	0.2),
("patched_files/Train", "augmented_train",              "augmentation", 	(0,0), 		 (), 		 	0.0), 
("Train_preproc", 		"kmeans_4", 					"kmeans", 			(0,0), 		 (), 		 	0.0),
("kmeans_4", 			"kmeans_4_centers", 			"nuclei_detection", (), 		 ('nucles_model_v3.meta', True, "kmeans_4_labels"), 0.0),
("kmeans_4", 			"image_nucleus", 				"image_to_nucleus", (48,48), 	 ("kmeans_4_centers", 1.0), 						0.2)
]




for action in actions:
	print("actions started for", action)
	folder_edit(action[0], action[1], action[2], action[3], action[4], action[5], random_seed)