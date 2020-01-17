import os
#for changing the keras background
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

imgW = 256
imgH = 256
numOfEpochs = 35
batch_size = 8
afterAugImageCount = 48
trainFolder = "patched256/Train"
validationFolder = "patched256/Validation"

import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import load_model
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot
from keras.optimizers import SGD

from keras.layers.normalization import BatchNormalization
import numpy as np

import pandas as pd
from keras import backend as K


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
		#rotation_range=40,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True,
        #fill_mode='nearest',
        rescale=1./255,
        )

# this is the augmentation configuration we will use for testing:
# only rescaling
valid_datagen = ImageDataGenerator(
	rescale=1./255
	)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        trainFolder,  # this is the target directory
        target_size=(imgW, imgH),  # all images will be resized to 150x150
        batch_size=batch_size,
        color_mode="rgb",
        class_mode='categorical',
        shuffle=True,
		seed=42)  

# this is a similar generator, for validation data
validation_generator = valid_datagen.flow_from_directory(
        validationFolder,
        target_size=(imgW, imgH),
        batch_size=batch_size,
        color_mode="rgb",
        class_mode='categorical',
        #class_mode=None,
        shuffle=True,
		seed=42)


opt = SGD(lr=0.01, momentum=0.5)

isExist = os.path.exists("./model.h5")
if(isExist):

	# load json and create model
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	# load weights into new model
	model.load_weights("model.h5")
	print("Loaded model from disk")

	model.summary()


	

	model.compile(loss="categorical_crossentropy",
	#optimizer can be changed or usable for default, check line 75
	#optimizer=opt,
	optimizer='rmsprop',
	metrics=['accuracy'])
else:

	var_padding = "valid"
	activation = "relu" #"relu"
	model = Sequential()

	model.add(Conv2D(32, (3, 3), padding='same', input_shape=(imgW, imgH,3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(3, 3)))

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
	model.add(Dense(256, activation='relu'))
	model.add(Dense(256, activation='relu'))
	model.add(Dense(4, activation='softmax'))

	#optimizer can be changed or usable for default, check line 75
	model.compile(loss="categorical_crossentropy",
	#optimizer=opt,
	optimizer='rmsprop',
	metrics=['accuracy'])

	model.summary()
	
	#saving the trained model
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
	
	#saving model on every epoch or only best
	filepath = "saved-model-{epoch:02d}-{val_acc:.2f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
	callbacks_list = [checkpoint]

	STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
	STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size
	model.fit_generator(
	        generator=train_generator,
	        callbacks=callbacks_list,
	        steps_per_epoch=STEP_SIZE_TRAIN,
	        epochs=numOfEpochs,
	        validation_data=validation_generator,
	        validation_steps=STEP_SIZE_VALID,
	        )
	
	# evaluate the model
	#scores = model.evaluate(X, Y, verbose=0)
	#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

	# serialize model to JSON
	#model_json = model.to_json()
	#with open("model.json", "w") as json_file:
	#		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("model.h5")
	print("Saved model to disk")
 

#Predict area
#this part of code is has not been checked in detail
#this part of code is has not been checked in detail
#this part of code is has not been checked in detail

"""
test_datagen = ImageDataGenerator(
	rescale=1./255
	)


test_generator = test_datagen.flow_from_directory(
	directory="Test_old_256x256",
	target_size=(imgW, imgH),
	batch_size=1,
	class_mode=None,
	shuffle=False,
	seed=42
)

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})

allPred = results.head(0);
rows = []

for i in range(0,test_generator.n,afterAugImageCount):
	subItems=results.iloc[i:i+afterAugImageCount] 

	symbols = subItems.groupby(['Predictions']).size().reset_index(name='counts').sort_values("counts",ascending = False).head(1)

	row = [i//afterAugImageCount, symbols.iat[0,0]]
	rows.append(row)

for i in rows:
	print(i[0],i[1])
"""