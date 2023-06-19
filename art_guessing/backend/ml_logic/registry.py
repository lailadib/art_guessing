# import glob
import os
from tensorflow import keras
from params import *

def save_model() -> None:


    return None


def load_model()  -> None:
    """
    Return a saved model
    """
    print("\n Load latest model from local registry...")

    art_model_path = os.path.join(LOCAL_MODEL_PATH, "efficientnet_v1.h5")
    art_model = keras.models.load_model(art_model_path)

    print("✅ Model loaded from local disk")

    print(art_model.summary())
    print(art_model_path)

    return art_model

load_model()
