import os
##################  CONSTANTS  #####################
LOCAL_DATA_PATH = "./raw_data"
LOCAL_MODEL_PATH = "./models"
STD_MODEL_NAME = "efficientnetb2_v2.h5"
IMG_SIZE = 256
INPUT_SHAPE = (int(IMG_SIZE),int(IMG_SIZE), 3)
CLASS_NAMES = ['art_nouveau',
    'baroque',
    'expressionism',
    'impressionism',
    'post_impressionism',
    'realism',
    'renaissance',
    'romanticism',
    'surrealism',
    'ukiyo_e'
    ]
