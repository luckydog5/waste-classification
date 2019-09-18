from resnet50 import ResNet50 
import os
import argparse
import numpy as np 
import tensorflow as tf 
from keras import backend as K 
from keras import layers
from keras.layers import Conv2D
from keras.layers.core import Dense,Dropout
from keras.layers import Flatten
from keras.models import Model 
from keras.optimizers import SGD,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.callbacks import EarlyStopping,ModelCheckpoint,LearningRateScheduler,TensorBoard,ReduceLROnPlateau


img_w = 224
img_h = 224
batch_size = 8

def gen(train_path):
	train_datagen = ImageDataGenerator(rotation_range=90,horizontal_flip=True,vertical_filp=True,fill_mode='nearest')
	train_generator = train_datagen.flow_from_directory(directory=train_path,target_size=(img_w,img_h),batch_size=batch_size,class_mode='categorical')


	return train_generator



def bulid_model(input_shape,dropout,fc_layers,num_classes):
	x = ResNet50(input_shape)
	x = Flatten()(x)
	for fc in fc_layers:
		x = Dense(fc,activation='relu')(x)
		#x = Dropout(dropout)(x)

	predictions = Dense(num_classes,activation='softmax')(x)
	model = Model(inputs=input_shape,outputs=predictions)
	model.summary()
	return model


def parse_arguments():

	parser = argparse.ArgumentParser(description='Some parameters.')
	
	parser.add_argument(
		"--train_path",
		type=str,
		help="Image path.",
		default=""
	)
	return parser.parse_args()


if __name__ == '__main__':

	input_shape = (img_h,img_w,3)
	dropout = 0.2
	fc_layers = [1024,1024]
	num_classes = 6
	epochs = 30
	args = parse_arguments()
	train_path = args.train_path
	train_generator = gen(train_path)
	model = bulid_model(input_shape=input_shape,dropout=dropout,fc_layers=fc_layers,num_classes=num_classes)

	for cls,idx in train_generator.class_indices.items():
		print('Class #{} = {}'.format(idx,cls))

	checkpoint = ModelCheckpoint(filepath='weights/weights-{epoch:03d}-{val_loss:.2f}.h5',monitor='val_loss',save_best_only=False,save_weights_only=True)

	checkpoint.set_model(model)

	model.compile(optimizer=Adam(lr=1e-5),loss='categorical_crossentropy',metrics=['accuracy'])

	lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),cooldown=0,patience=5,min_lr=0.5e-6)

	earlystopping = EarlyStopping(monitor='val_loss',patience=5,verbose=1)

	tensorbord = TensorBord(log_dir='weights/logs',write_graph=True)

	model.fit_generator(generator=train_generator,steps_per_epoch=1000,epochs=epochs,initial_epoch=0,callbacks=[checkpoint,lr_reducer,earlystopping,tensorbord])






























