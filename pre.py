from keras.preprocessing import image

from train import bulid_model
import os
import glob
import numpy as np 

if __name__ == '__main__':

	img_h = 224
	img_w = 224
	input_shape = (224,224,3)
	fc_layers = [1024,1024]
	num_classes = 6
	image_path = 'test/'
	weights_path = 'weights/weights-030-0.01.h5'
	images = glob.glob(image_path+'*.jpg')
	cls_list = ['cardboard','glass','metal','paper','plastic','trash']
	model = bulid_model(input_shape=input_shape,dropout=0,fc_layers=fc_layers,num_classes=num_classes)
	model.load_weights(weights_path)
	for f in images:
		img = image.load_img(f,target_size=(img_h,img_w))
		if img is None:
			continue

		x = image.img_to_array(img)
		x = np.expand_dims(x,axis=0)
		pred = model.predict(x)[0]
		top_inds = pred.argsort()[::-1][:5]
		print(f)
		for i in top_inds:
			print(' {:.3f}  {}'.format(pred[i],cls_list[i]))
