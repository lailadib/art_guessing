from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow import convert_to_tensor
import os
from params import *

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
    img_size = (256, 256)
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

def preprocess_new_image():
    """
    Preprocessing the uploaded image.
    It will be cropped to a square, then resized to become a 256x256 image
    Load this image as a tensor object to be able to predict with it
    """
    ### Crop and resize the image

    ### Load the image as a tensor



train_ds, vals_ds, test_ds = images_to_dataset()
for image_batch, labels_batch in train_ds :
    print(image_batch.shape)
    print(labels_batch.shape)
    break
