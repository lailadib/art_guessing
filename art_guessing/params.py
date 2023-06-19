# params.py contains the project's global variables/parameters (including variables from .env)
import os

##################  VARIABLES  ##################
MODEL_TARGET = os.environ.get("MODEL_TARGET")    # get MODEL_TARGET from .env (e.g. local, gcs, mflow)
GCP_PROJECT = os.environ.get("GCP_PROJECT")      # get GCP_PROJECT from .env (personal GCP project for this bootcamp)
BUCKET_NAME = os.environ.get("BUCKET_NAME")      # get BUCKET_NAME from .env (cloud storage)
##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "code", "lailadib", "art_guessing", "art_guessing", "raw_data", "artbench_10")
LOCAL_MODEL_PATH =  os.path.join(os.path.expanduser('~'), "code", "lailadib", "art_guessing", "art_guessing", "models")

IMG_SIZE = os.environ.get("IMG_SIZE")
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
