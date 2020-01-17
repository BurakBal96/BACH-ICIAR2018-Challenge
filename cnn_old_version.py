import os

#for changing the keras background
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import cv2 as cv
import keras
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
from keras.models import load_model

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.layers.normalization import BatchNormalization

labels = ['Benign','InSitu','Invasive','Normal']
input_shape, train_labels, train_images, test_labels, test_images = [0,0,0,0,0]

batch_size = 16
epochs = 50
num_classes = 4

seed = 7
np.random.seed(7)

imgW = 256
imgH = 256

afterAugImageCount = 48

folderName = "patched256/Train"
validationFolderName = "patched256/Validation"
predict_folder = "predictImages"

imgTypes = os.listdir(folderName)
isExist = os.path.exists("./model.h5")

#loading the saved dataset
isExist = os.path.exists("./saves_test_images.npy")
img_data = []
if(isExist):
	test_images = np.load('saves_test_images.npy')
	test_labels = np.load('saved_test_labels.npy')

	train_images = np.load('saved_train_images.npy')
	train_labels = np.load('saved_train_labels.npy')

	input_shape = np.load('saved_input_shape.npy')
	labels = np.load('saved_labels.npy')

	print('npy loaded')
else:
	img_data = []

	####### reading train image names
	imgPaths = []
	for i in range(0, len(imgTypes)):
		imgTypePath = folderName + "/" + imgTypes[i]
		print(imgTypePath)
		klasor = os.listdir(imgTypePath)
		for j in range(0, len(klasor)):
			imgPaths.append(imgTypePath + "/" + klasor[j])
			#print(os.listdir(imgTypePath)[j])

	print('dosya isimleri okundu')

	#Reading all files into a list
	img_data_list=[]
	for i in range (0, len(imgPaths)):
		im = cv.imread(imgPaths[i])
		im = cv.resize(im, (imgH,imgW))
		img_data_list.append(im)

	print("dosyalar okundu")
	img_data = np.array(img_data_list)

	# Define the number of samples
	num_of_samples = len(img_data)
	labels = np.ones((num_of_samples,),dtype='int64')

	start = 0

	for i in range(0,len(imgTypes)):
		imgTypePath = folderName + "/" + imgTypes[i]
		stop = len(os.listdir(imgTypePath)) + start
		labels[start:stop] = i
		start = stop



	# convert class labels to on-hot encoding
	Y = np_utils.to_categorical(labels, num_classes)

	#Shuffle the dataset
	train_images,train_labels = shuffle(img_data,Y, random_state=seed)
	####### end of reading train images


	####### reading validation image names
	imgPaths = []
	for i in range(0, len(imgTypes)):
		imgTypePath = validationFolderName + "/" + imgTypes[i]
		print(imgTypePath)
		klasor = os.listdir(imgTypePath)
		for j in range(0, len(klasor)):
			imgPaths.append(imgTypePath + "/" + klasor[j])
			#print(os.listdir(imgTypePath)[j])

	print('dosya isimleri okundu')
	#Reading all files into a list
	img_data_list=[]
	for i in range (0, len(imgPaths)):
		im = cv.imread(imgPaths[i])
		im = cv.resize(im, (imgH,imgW))
		img_data_list.append(im)

	print("dosyalar okundu")
	img_data = np.array(img_data_list)

	# Define the number of samples
	num_of_samples = len(img_data)
	labels = np.ones((num_of_samples,),dtype='int64')

	start = 0

	for i in range(0,len(imgTypes)):
		imgTypePath = validationFolderName + "/" + imgTypes[i]
		stop = len(os.listdir(imgTypePath)) + start
		labels[start:stop] = i
		start = stop

	# convert class labels to on-hot encoding
	Y = np_utils.to_categorical(labels, num_classes)

	#Shuffle the dataset
	test_images,test_labels = shuffle(img_data,Y, random_state=seed)
	####### end of reading validation images


	input_shape = img_data[0].shape

	np.save('saves_test_images.npy', test_images)
	np.save('saved_test_labels.npy',test_labels)

	np.save('saved_train_images.npy', train_images)
	np.save('saved_train_labels.npy',train_labels)

	np.save('saved_input_shape.npy', input_shape)
	np.save('saved_labels.npy', labels)

	print('npy saved')
	del img_data

	# Split the dataset
	#train_images, test_images, train_labels, test_labels = train_test_split(img_data, Y, test_size=0.2, random_state=seed)


#data normalization
train_images = train_images.astype('float16')
train_images /= 255

test_images = test_images.astype('float16')
test_images /= 255


	  
		
var_padding = "valid"
activation = "relu"
print(input_shape)
	
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(imgW, imgH,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
#model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.3))


model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
#model.add(Dense(64, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))


#optimizer can be changed or usable for default
optimizer = SGD(lr=0.005)
model.compile(loss=keras.losses.categorical_crossentropy,
#optimizer=opt,
optimizer='rmsprop',
metrics=['accuracy'])
model.summary()


#checkpoint save and reducing learning rate
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stop = EarlyStopping(monitor='val_loss',patience=10)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=6, verbose=1, mode='auto',
                              min_delta=0.0001, cooldown=0, min_lr=1e-8)

filepath = "saved-model-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint,
# early_stop, 
#reduce_lr
]
# CheckPoint TestAlani sonu



#loading model from disk
isExist = os.path.exists("./model.hdf5")
if(isExist):
	model.load_weights("model.hdf5")
else:
	hist = model.fit(train_images, train_labels,
		callbacks=callbacks_list, 
		batch_size=batch_size, 
		nb_epoch=epochs, 
		verbose=1,
		validation_data=(test_images, test_labels)
		)



### predict the model
score = model.evaluate(test_images, test_labels, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

if(not isExist):
	# serialize model to JSON
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
	  json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("model.h5")
	print("Saved model to disk")
	model.save('model.h5')


import pandas as pd
labels = ['Benign','InSitu','Invasive','Normal']
imgPaths = []

imgPaths = os.listdir(predict_folder)

for i in range(0,len(imgPaths)):
	imgPaths[i] = predict_folder+"/" + imgPaths[i]



#Reading all files into a list
img_data_list=[]
for i in range (0, len(imgPaths)):
	im = cv.imread(imgPaths[i])
	im = cv.resize(im, (imgH,imgW))
	img_data_list.append(im)
	im = np.array(im)






img_data = np.array(img_data_list)
img_data = img_data.astype('float16')
img_data /= 255


classes = model.predict_classes(img_data, batch_size=1)
#for i in range(0,len(classes)):
#	print(labels[classes[i]]+"\t",classes[i],"\t"+imgPaths[i][imgPaths[i].rfind('/')+1:])



results=pd.DataFrame({"Predictions":classes})

print(results)
rows = []

for i in range(0,1728,afterAugImageCount):
	subItems=results.iloc[i:i+afterAugImageCount] 

	symbols = subItems.groupby(['Predictions']).size().reset_index(name='counts').sort_values("counts",ascending = False).head(1)

	row = [i//afterAugImageCount, symbols.iat[0,0]]
	rows.append(row)

for i in rows:
	print(i[0],labels[i[1]])