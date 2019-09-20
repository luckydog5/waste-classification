# waste-classification

This is a simple image classification repositoty for waste images with six classes.

['cardboard','glass','metal','paper','plastic','trash'] each class contains several hundred of images. 

You can download the dataset here: https://github.com/garythung/trashnet/blob/master/data/dataset-resized.zip

Since this is an image classification task, i fine-tuned the pre-trained ResNet50 model.

First,I remove the origin classification layers in ResNet50

Then, i add two fc-layers.

Finally, i add one classification layers with output classes six.

# Train

Cause i have a small amount of training samples about 2 thound  images, i implement data augmentation during training time.

Here i used three augmentation methods: rotation, horizontal_flip and vertical_flip, and these are easy implemented with keras 

"ImageDataGenerator" method.

python train.py --train_path path_to_your_images

I haven't implemented validation during the training time,you can implement it by yourself if you like.


