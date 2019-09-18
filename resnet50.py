from __future__ import print_function
import numpy as np 
import warnings
from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model 
import keras.backend as K 


def identi_block(input_tensor,kernel_size,filters,stage,block):
	"""The identity block is the block that has no conv layer at shortcut.

	# Arguments
		input_tensor: input tensor
		kernel_size: defualt 3, the kernel size of middle conv layer at main path
		filters: list of integers, the filterss of 3 conv layer at main path
		stage: integer, current stage label, used for generating layer names
		block: 'a','b'..., current block label, used for generating layer names

	# Returns
		Output tensor for the block.
	"""
	filters1,filters2,filters3 = filters
	if K.image_data_format() == 'channels_last':
		bn_axis = 3

	else:
		bn_axis = 1

	# conv and bn layer's name
	conv_name_base = 'res' + str(stage) + block + '_branch'

	bn_name_base = 'bn' + str(stage) + block + '_branch'


	x = Conv2D(filters1,(1,1),name=conv_name_base+'2a')(input_tensor)
	x = BatchNormalization(axis=bn_axis,name=bn_name_base+'2a')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters2,kernel_size,padding='same',name=conv_name_base+'2b')(x)
	x = BatchNormalization(axis=bn_axis,name=bn_name_base+'2b')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters3,(1,1),name=conv_name_base+'2c')(x)
	x = BatchNormalization(axis=bn_axis,name=bn_name_base+'2c')(x)

	x = layers.add([x,input_tensor])
	x = Activation('relu')(x)

	return x

def conv_block(input_tensor,kernel_size,filters,stage,block,strides=(2,2)):
	"""conv_block is the block that has a conv layer at shortcut

	# Arguments
		input_tensor: input tensor
		kernel_size: defualt 3, the kernel size of middle conv layer at main path
		filters: list of integers, the filterss of 3 conv layer at main path
		stage: integer, current stage label, used for generating layer names
		block: 'a','b'..., current block label, used for generating layer names

	# Returns
		Output tensor for the block.

	Note that from stage 3, the first conv layer at main path is with strides=(2,2)
	And the shortcut should have strides=(2,2) as well
	"""

	filters1,filters2,filters3 = filters
	if K.image_data_format() == 'channels_last':
		bn_axis = 3

	else:
		bn_axis = 1


	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Conv2D(filters1,(1,1),strides=strides,name=conv_name_base+'2a')(input_tensor)
	x = BatchNormalization(axis=bn_axis,name=bn_name_base+'2a')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters2,kernel_size,padding='same',name=conv_name_base+'2b')(x)
	x = BatchNormalization(axis=bn_axis,name=bn_name_base+'2b')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters3,(1,1),name=conv_name_base+'2c')(x)
	x = BatchNormalization(axis=bn_axis,name=bn_name_base+'2c')(x)

	shortcut = Conv2D(filters3,(1,1),strides=strides,name=conv_name_base+'1')(input_tensor)
	shortcut = BatchNormalization(axis=bn_axis,name=bn_name_base+'1')(shortcut)

	x = layers.add([x,shortcut])
	x = Activation('relu')(x)
	return x

def ResNet50(input_tensor=None):
	"""
	Input: input image shape e.g.(224,224,3)

	return: tensor(1,1,2048)

	"""

	if K.image_data_format() == 'channels_last':
		bn_axis = 3
	else:
		bn_axis = 1

	x = ZeroPadding2D((3,3))(input_tensor)
	x = Conv2D(64,(7,7),strides=(2,2),name='conv1')(x)
	x = BatchNormalization(axis=bn_axis,name='bn_conv1')(x)
	x = Activation('relu')(x)
	x = MaxPooling2D((3,3),strides=(2,2))(x)
	# w = w/4      h = h/4
	x = conv_block(x,3,[64,64,256],stage=2,block='a',strides=(1,1))
	x = identi_block(x,3,[64,64,256],stage=2,block='b')
	x = identi_block(x,3,[64,64,256],stage=2,block='c')
	## no change
	x = conv_block(x,3,[128,128,512],stage=3,block='a')
	x = identi_block(x,3,[128,128,512],stage=3,block='b')
	x = identi_block(x,3,[128,128,512],stage=3,block='c')
	x = identi_block(x,3,[128,128,512],stage=3,block='d')
	# w = w/8      h = h/8
	x = conv_block(x,3,[256,256,1024],stage=4,block='a')
	x = identi_block(x,3,[256,256,1024],stage=4,block='b')
	x = identi_block(x,3,[256,256,1024],stage=4,block='c')
	x = identi_block(x,3,[256,256,1024],stage=4,block='d')
	x = identi_block(x,3,[256,256,1024],stage=4,block='e')
	x = identi_block(x,3,[256,256,1024],stage=4,block='f')
	# w = w/16.      h=h/16
	x = conv_block(x,3,[512,512,2048],stage=5,block='a')
	x = identi_block(x,3,[512,512,2048],stage=5,block='b')
	x = identi_block(x,3,[512,512,2048],stage=5,block='c')
	## w/=32.   h/=32.      (7,7,2048). ---->>>>(224,224,3)

	x = AveragePooling2D((7,7),name='avg_pool')(x)

	return x












