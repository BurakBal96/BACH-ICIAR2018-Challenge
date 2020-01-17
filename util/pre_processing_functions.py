import cv2
import numpy as np
import os
from PIL import Image
from PIL import ImageOps
from joblib import Parallel, delayed
import random
from sklearn import cluster
from matplotlib.pyplot import imread, waitforbuttonpress
import math
from skimage.morphology import square, erosion, dilation
from skimage.measure import label, regionprops
import dill as pickle
import matplotlib.pyplot as plt
import tensorflow as tf 



centers = np.array([ [122,28,45],
	[231,214,211],
	[209,157,187],
	[171,70,168] 
	])

def create_folder(folder_name):
	if(not os.path.exists(folder_name)):
		os.mkdir(folder_name)

def kmeans(input_image_name, output_image_name, function_args):
	image = cv2.imread(input_image_name)
	x, y, z = image.shape
	image_2d = image.reshape(x*y, z)

	kmeans_clusters = len(centers)

	kmeans_cluster = cluster.KMeans(n_clusters=kmeans_clusters, init=centers,n_init=1, max_iter=10)
	kmeans_cluster.fit(image_2d)

	cluster_centers = kmeans_cluster.cluster_centers_
	cluster_labels = kmeans_cluster.labels_
	for index in range(0,len(cluster_labels)):
		if(cluster_labels[index] != 0):
			image_2d[index] = [255, 255, 255] ## all zeros are black, 255s are white

	#save img
	result_img = image_2d.reshape(image.shape)
	cv2.imwrite(output_image_name, result_img)
	

def cropping(input_image_name, output_image_name, patch_sizes, function_args):
	step_sizes = function_args
	image = cv2.imread(input_image_name)

	output_image_name_splitted = os.path.splitext(output_image_name)

	for x in range(0, image.shape[0]-patch_sizes[0]+1, step_sizes[0]):
		for y in range(0, image.shape[1]-patch_sizes[1]+1, step_sizes[1]):
			crop_img = image[x:x+patch_sizes[0], y:y+patch_sizes[1]]
		
			output_img_name_ = output_image_name_splitted[0] + "_" +str(x)+"x"+ str(y) + output_image_name_splitted[1]
			cv2.imwrite(output_img_name_, crop_img)
		


def augmentation(input_image_name, output_image_name):
	image = Image.open(input_image_name)
	image.save(output_image_name) #save original image

	output_image_name_splitted = os.path.splitext(output_image_name)

	image.save(output_image_name_splitted[0] + output_image_name_splitted[1])

	edited_image = image.rotate(90)
	edited_image.save(output_image_name_splitted[0] + "_90" + output_image_name_splitted[1])

	edited_image = image.rotate(180)
	edited_image.save(output_image_name_splitted[0] + "_180" + output_image_name_splitted[1])

	edited_image = image.rotate(270)
	edited_image.save(output_image_name_splitted[0] + "_270" + output_image_name_splitted[1])

	edited_image = ImageOps.mirror(image)
	edited_image.save(output_image_name_splitted[0] + "_mirror" + output_image_name_splitted[1])


def image_to_nucleus(input_image_name, output_image_name, patch_sizes, centers_path, nuclei_rate):
	patchH_half = int(patch_sizes[0]/2)
	patchW_half = int(patch_sizes[1]/2)
	image = cv2.imread(input_image_name)

	centers = np.loadtxt(centers_path)
	centers_shuffle = np.random.shuffle(centers)
	
	for center in centers[0:int(len(centers)*nuclei_rate)]:
		center_int = np.int32(center)

		x_start = center_int[0] - patchH_half
		if(x_start < 0 ):
			x_start = 0

		y_start = center_int[1] - patchH_half
		if(y_start < 0 ):
			y_start = 0  

		x_end = center_int[0] + patchH_half	
		if(x_end > image.shape[0]-1):
			x_end = image.shape[0]-1

		y_end = center_int[1] + patchH_half	
		if(y_end > image.shape[1]-1):
			y_end = image.shape[1]-1
		#crop
		imageNew = image[x_start:x_end, y_start:y_end]
		imageNew = cv2.resize(imageNew,(patch_sizes[0], patch_sizes[1]))

		#save img
		output_path_splitted = os.path.splitext(output_image_name)
		save_path = output_path_splitted[0] + "_"+str(x_start)+"x"+str(y_start) + output_path_splitted[1]
		#print(save_path)
		cv2.imwrite(save_path,imageNew)
	


'''
#############################################################
utility functions assisting nuclei detection and segmentation
@author: Kemeng Chen
@edited by Burak Bal
'''
class restored_model(object):

	def __init__(self, model_name, model_folder):
		self.graph=tf.Graph()
		self.sess=tf.Session(graph=self.graph)
		print('Read model: ', model_name)

		with self.graph.as_default():
			self.model_saver=tf.train.import_meta_graph(model_name)
			self.model_saver.restore(self.sess, tf.train.latest_checkpoint(model_folder+'/.'))
			self.graph=self.graph
			self.sample_in=self.graph.get_tensor_by_name('sample:0')
			self.c_mask_out=self.graph.get_tensor_by_name('c_mask:0')

	def run_sess(self, patches):
		feed_dict={self.sample_in: patches}
		generated_mask=self.sess.run([self.c_mask_out], feed_dict)
		return generated_mask

	def close_sess(self):
		self.sess.close()

def print_ctime():
	current_time=ctime(int(time()))
	print(str(current_time))

def batch2list(batch):
	mask_list=list()
	for index in range(batch.shape[0]):
		mask_list.append(batch[index,:,:])
	return mask_list

def patch2image(patch_list, patch_size, stride, shape):	
	if shape[0]<patch_size:
		L=0
	else:
		L=math.ceil((shape[0]-patch_size)/stride)
	if shape[1]<patch_size:
		W=0
	else:
		W=math.ceil((shape[1]-patch_size)/stride)	

	full_image=np.zeros([L*stride+patch_size, W*stride+patch_size])
	bk=np.zeros([L*stride+patch_size, W*stride+patch_size])
	cnt=0
	for i in range(L+1):
		for j in range(W+1):
			full_image[i*stride:i*stride+patch_size, j*stride:j*stride+patch_size]+=patch_list[cnt]
			bk[i*stride:i*stride+patch_size, j*stride:j*stride+patch_size]+=np.ones([patch_size, patch_size])
			cnt+=1   
	full_image/=bk
	image=full_image[0:shape[0], 0:shape[1]]
	return image

def image2patch(in_image, patch_size, stride, blur=False, f_size=9):
	if blur is True:
		in_image=cv2.medianBlur(in_image, f_size)
		# in_image=denoise_bilateral(in_image.astype(np.float), 19, 11, 9, multichannel=False)
	shape=in_image.shape
	if shape[0]<patch_size:
		L=0
	else:
		L=math.ceil((shape[0]-patch_size)/stride)
	if shape[1]<patch_size:
		W=0
	else:
		W=math.ceil((shape[1]-patch_size)/stride)	
	patch_list=list()
	
	if len(shape)>2:
		full_image=np.pad(in_image, ((0, patch_size+stride*L-shape[0]), (0, patch_size+stride*W-shape[1]), (0,0)), mode='symmetric')
	else:
		full_image=np.pad(in_image, ((0, patch_size+stride*L-shape[0]), (0, patch_size+stride*W-shape[1])), mode='symmetric')
	for i in range(L+1):
		for j in range(W+1):
			if len(shape)>2:
				patch_list.append(full_image[i*stride:i*stride+patch_size, j*stride:j*stride+patch_size, :])
			else:
				patch_list.append(full_image[i*stride:i*stride+patch_size, j*stride:j*stride+patch_size])
	if len(patch_list)!=(L+1)*(W+1):
		raise ValueError('Patch_list: ', str(len(patch_list), ' L: ', str(L), ' W: ', str(W)))
	
	return patch_list

def list2batch(patches):
	'''
	covert patch to flat batch
	args:
		patches: list
	return:
		batch: numpy array
	'''
	patch_shape=list(patches[0].shape)

	batch_size=len(patches)
	
	if len(patch_shape)>2:
		batch=np.zeros([batch_size]+patch_shape)
		for index, temp in enumerate(patches):
			batch[index,:,:,:]=temp
	else:
		batch=np.zeros([batch_size]+patch_shape+[1])
		for index, temp in enumerate(patches):
			batch[index,:,:,:]=np.expand_dims(temp, axis=-1)
	return batch

def preprocess(input_image, patch_size, stride):
	f_size=5
	g_size=10
	shape=input_image.shape
	patch_list=image2patch(input_image.astype(np.float32)/255.0, patch_size, stride)
	num_group=math.ceil(len(patch_list)/g_size)
	batch_group=list()
	for i in range(num_group):
		temp_batch=list2batch(patch_list[i*g_size:(i+1)*g_size])
		batch_group.append(temp_batch)
	return batch_group, shape

def sess_interference(sess, batch_group):
	patch_list=list()
	for temp_batch in batch_group:
		mask_batch=sess.run_sess(temp_batch)[0]
		mask_batch=np.squeeze(mask_batch, axis=-1)
		mask_list=batch2list(mask_batch)
		patch_list+=mask_list
	return patch_list

def center_point(mask, file_name, output_txt):
	v,h=mask.shape
	center_mask=np.zeros([v,h])
	mask=erosion(mask, square(3))
	individual_mask=label(mask, connectivity=2)
	prop=regionprops(individual_mask)
	
	center_list = []
	for cordinates in prop:
		temp_center=cordinates.centroid
		center_list.append(temp_center)
		
		if not math.isnan(temp_center[0]) and not math.isnan(temp_center[1]):
			temp_mask=np.zeros([v,h])
			temp_mask[int(temp_center[0]), int(temp_center[1])]=1
			center_mask+=dilation(temp_mask, square(2))

	new_file_name = os.path.splitext(output_txt)[0] +"_centers.txt"
	centers = np.array(center_list)
	np.savetxt(new_file_name, centers)

	

	return np.clip(center_mask, a_min=0, a_max=1).astype(np.uint8)

def draw_individual_edge(mask):
	v,h=mask.shape
	edge=np.zeros([v,h])
	individual_mask=label(mask, connectivity=2)
	for index in np.unique(individual_mask):
		if index==0:
			continue
		temp_mask=np.copy(individual_mask)
		temp_mask[temp_mask!=index]=0
		temp_mask[temp_mask==index]=1
		temp_mask=dilation(temp_mask, square(3))
		temp_edge=cv2.Canny(temp_mask.astype(np.uint8), 2,5)/255
		edge+=temp_edge
	return np.clip(edge, a_min=0, a_max=1).astype(np.uint8)

def center_edge(mask, image, img_name, with_label, output_txt):
	center_map=center_point(mask, img_name, output_txt)
	if(with_label):
		edge_map=draw_individual_edge(mask)
		comb_mask=center_map+edge_map
		comb_mask=np.clip(comb_mask, a_min=0, a_max=1)
		check_image=np.copy(image)
		comb_mask*=255
		check_image[:,:,1]=np.maximum(check_image[:,:,1], comb_mask)
		return check_image.astype(np.uint8), comb_mask.astype(np.uint8)
	return 0,0


def process_for_one_img(img_name, output_txt, model, patch_size, stride, with_label, output_image):
	temp_image = cv2.imread(img_name)

	batch_group, shape=preprocess(temp_image, patch_size, stride)
	mask_list=sess_interference(model, batch_group)
	c_mask=patch2image(mask_list, patch_size, stride, shape)
	c_mask=cv2.medianBlur((255*c_mask).astype(np.uint8), 3)
	c_mask=c_mask.astype(np.float)/255
	thr=0.5
	c_mask[c_mask<thr]=0
	c_mask[c_mask>=thr]=1
	center_edge_mask, gray_map = center_edge(c_mask, temp_image, img_name, with_label, output_txt)



	if(with_label):
		img_name_with_ex = os.path.splitext(output_image)
		mask_name = img_name_with_ex[0] +"_mask" + img_name_with_ex[1]
		label_name = img_name_with_ex[0] +"_label" + img_name_with_ex[1]

		cv2.imwrite(mask_name, gray_map)
		cv2.imwrite(label_name, center_edge_mask)

def nuclei_detection_process(data_folder, output_folder, model_name, with_label, label_folder_name):
	patch_size=128
	stride=64
	
	class_names = os.listdir(data_folder)
	img_paths = []

	if(with_label):
		create_folder(label_folder_name)

	for i in range(0, len(class_names)):
		imgTypePath = data_folder + "/" + class_names[i]

		if(with_label):
			create_folder(label_folder_name + "/"+class_names[i])

		folder = os.listdir(imgTypePath)
		for j in range(0, len(folder)):
			img_paths.append(imgTypePath + "/" + folder[j])

	model_path=os.path.join(os.getcwd(), 'models')
	model=restored_model(os.path.join(model_path, model_name), model_path)

	for temp_path in img_paths:
		print("processing :",temp_path)
		output_image = temp_path.replace(data_folder,label_folder_name, 1)
		output_txt = temp_path.replace(data_folder, output_folder, 1)
		process_for_one_img(temp_path, output_txt, model, patch_size, stride, with_label, output_image)
		
	model.close_sess()

'''
End of Kemeng Chen's nuclei detection code
##########################################

'''


def folder_edit(input_folder_name, output_folder_name, function_name, patch_sizes, function_args, validation_size, random_seed):
	labels = os.listdir(input_folder_name)

	input_train_images = []
	input_validation_images = []
	output_train_images = []
	output_validation_images = []

	if(validation_size > 0):
		output_train_folder_name = output_folder_name + "/Train/"
		output_validation_folder_name = output_folder_name + "/Validation/"
	else:
		output_train_folder_name = output_validation_folder_name = output_folder_name + "/"

	create_folder(output_folder_name)
	create_folder(output_train_folder_name)
	create_folder(output_validation_folder_name)

	for label in labels:
		input_folder_name_label = input_folder_name + "/" + label
		folder = os.listdir(input_folder_name_label)
		random.Random(random_seed).shuffle(folder)

		output_train_folder_name_label = output_train_folder_name + label
		output_validation_folder_name_label = output_validation_folder_name+ label

		create_folder(output_train_folder_name_label)
		create_folder(output_validation_folder_name_label)

		train_size = int(len(folder)*(1.0-validation_size))
		for j in range(0, train_size):
			input_train_images.append(input_folder_name_label + "/" + folder[j])
			output_train_images.append(output_train_folder_name_label + "/" + folder[j])
		for j in range(train_size, len(folder)):
			input_validation_images.append(input_folder_name_label + "/" + folder[j])
			output_validation_images.append(output_validation_folder_name_label + "/" + folder[j])


	if(function_name == "patch"):
		Parallel(n_jobs=-1)(delayed(cropping)(input_train_images[i], output_train_images[i], patch_sizes, function_args ) for i in range(0,len(input_train_images)))
		Parallel(n_jobs=-1)(delayed(cropping)(input_validation_images[i], output_validation_images[i], patch_sizes, function_args ) for i in range(0,len(input_validation_images)))
	
	elif(function_name == "augmentation"):
		Parallel(n_jobs=-1)(delayed(augmentation)(input_train_images[i], output_train_images[i] ) for i in range(0,len(input_train_images)))
		#augmentation is for trainset, not for validation
	
	elif(function_name == "kmeans"):
		Parallel(n_jobs=-1)(delayed(kmeans)(input_train_images[i], output_train_images[i], function_args ) for i in range(0,len(input_train_images)))
		Parallel(n_jobs=-1)(delayed(kmeans)(input_validation_images[i], output_validation_images[i], function_args ) for i in range(0,len(input_validation_images)))
	
	elif(function_name == "nuclei_detection"):
		nuclei_detection_process(input_folder_name, output_folder_name, function_args[0], function_args[1], function_args[2])
	
	elif(function_name == "image_to_nucleus"):
		train_center_paths = []
		validation_center_paths = []
		for label in labels:
			center_with_label = function_args[0] + "/" + label
			folder = os.listdir(center_with_label)

			train_size = int(len(folder)*(1.0-validation_size))
			for j in range(0, train_size):
				train_center_paths.append(center_with_label + "/" + folder[j])
				
			for j in range(train_size, len(folder)):
				validation_center_paths.append(center_with_label + "/" + folder[j])

		Parallel(n_jobs=-1)(delayed(image_to_nucleus)(input_train_images[i], output_train_images[i], patch_sizes, train_center_paths[i], function_args[1] ) for i in range(0,len(input_train_images)))
		Parallel(n_jobs=-1)(delayed(image_to_nucleus)(input_validation_images[i], output_validation_images[i], patch_sizes, validation_center_paths[i], function_args[1] ) for i in range(0,len(input_validation_images)))
		


