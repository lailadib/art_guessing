from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow import convert_to_tensor
import os
from params import *
from PIL import Image
import numpy as np

def images_to_dataset():
    """
    Transform the jpg contained in train and test folder
    in 3 datasets for training, validation and test
    Returns 3 tf.data.Dataset
    """
    #Image folders
    train_dir = os.path.join(LOCAL_DATA_PATH, 'train')
    test_dir = os.path.join(LOCAL_DATA_PATH, 'test')

    #Specify image size and batch size parameters
    img_size = (IMG_SIZE, IMG_SIZE)
    batch_size = 32

    #Create the datasets
    train_ds = image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset='training',
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        seed=42
    )
    val_ds = image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset='validation',
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        seed=42
    )
    test_ds = image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True
    )
    return train_ds, val_ds, test_ds

def preprocess_new_image(uploaded_file=None, to_crop_resize=False, padding=0, image_size=IMG_SIZE):
    """
    Preprocesses the uploaded image file wich is coming as jpeg.
    Returns a tensorflow tensor in the shape = (1, image_size, image_size, 3)
    The image will be cropped to a square then resized to become
    an of size image_size x image_size on request by setting to_crop_resize to True.
    By default the image is expected to be delievered already cropped to square
    and dowsized to image_size x image_size

    """

    if uploaded_file != None:
        image_size = image_size
        img = Image.open(uploaded_file)
        if to_crop_resize == True:
            #crop and downsize the image
            width, height = img.size
            diff = abs(width - height)
            if width != height: #crop if image is not square
                if width > height:
                    if padding >= height/2: padding = 0 #set padding 0 to prevent cut of whole image
                    l = diff/2 + padding
                    r = l + height - 2*padding
                    t = 0 + padding
                    b = height - padding
                elif width < height:
                    if padding >= width/2: padding = 0 #set padding 0 to prevent cut of whole image
                    l = 0 + padding
                    r = width - padding
                    t = diff/2 + padding
                    b = t + width - 2*padding
                img = img.crop((l,t,r,b))
            if image_size < img.size[0] and image_size < img.size[1]:
                img = img.resize((image_size, image_size))

        ### convert the image to a tensor
        img_tens = convert_to_tensor(img, dtype=np.float32)
        np.expand_dims(img_tens, axis=0)
        assert img_tens.shape == (1, image_size, image_size, 3)
        #img_tens.shape(1, image_size, image_size, 3)

    return img_tens

train_ds, vals_ds, test_ds = images_to_dataset()
for image_batch, labels_batch in train_ds :
    print(image_batch.shape)
    print(labels_batch.shape)
    break
