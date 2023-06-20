from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow import convert_to_tensor
from os.path import join
from params import *
from PIL import Image
import numpy as np

def images_to_dataset():
    """Transform the jpg contained in train and test folders
    in 3 datasets for training, validation and test.
    The validation set represents 20% of the training set.
    ----------
    Returns 3 tf.data.Dataset:
    train_ds, val_ds, test_ds.
    train_ds and val_ds contains (images, labels)
    test_ds contains (images)
    """
    #Image folders
    train_dir = join(LOCAL_DATA_PATH, 'train')
    test_dir = join(LOCAL_DATA_PATH, 'test')

    #Specify image size and batch size parameters
    img_size = (int(IMG_SIZE), int(IMG_SIZE))
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
    print("✅ Train, Validation and Test datasets ready to use")

    return train_ds, val_ds, test_ds

def preprocess_new_image(uploaded_file=None, to_crop_resize=False, padding=0, image_size=IMG_SIZE):
    """Preprocess the uploaded image file wich is coming as jpeg.
    The image will be cropped to a square then resized to become
    size (image_size x image_size) on request by setting to_crop_resize to True.
    By default the image is expected to be delivered already cropped to square
    and downsized to (image_size x image_size)
    ----------
    Arguments:
    uploaded_file -- path to the image
    to_crop_resize -- bool. False by default. Set to True if crop and resize in needed
    padding -- integer value. 0 by default.
    image_size -- integer. 256 by default. Size of the squared output image.
    ----------
    Returns a tensor with shape = (1, image_size, image_size, 3)
    """

    if uploaded_file != None:
        image_size = int(IMG_SIZE)
        img = Image.open(uploaded_file)

        if to_crop_resize == True:
            #Crop and downsize the image
            width, height = img.size
            diff = abs(width - height)

            if width != height: #Crop if image is not square
                if width > height:
                    if padding >= height/2: padding = 0 #Set padding 0 to prevent cut of whole image
                    l = diff/2 + padding
                    r = l + height - 2*padding
                    t = 0 + padding
                    b = height - padding
                elif width < height:
                    if padding >= width/2: padding = 0 #Set padding 0 to prevent cut of whole image
                    l = 0 + padding
                    r = width - padding
                    t = diff/2 + padding
                    b = t + width - 2*padding
                img = img.crop((l,t,r,b))
            if image_size < img.size[0] and image_size < img.size[1]:
                img = img.resize((image_size, image_size))

        #Convert the image to a tensor
        img_tens = convert_to_tensor(img, dtype=np.float32)
        img_tens = np.expand_dims(img_tens, axis=0)
        assert img_tens.shape == (1, image_size, image_size, 3)

    return img_tens
