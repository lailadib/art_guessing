# import glob
import os
from tensorflow import keras
from art_guessing.params import *

def save_model() -> None:


    return None


def load_model()  -> None:
    """
    Return a saved model
    """
    print("\n Load latest model from local registry...")

    models_path = LOCAL_MODEL_PATH
    art_model_path = os.path.join(models_path, "efficientnetb2_v2.h5")
    art_model = keras.models.load_model(art_model_path)

    print("✅ Model loaded from local disk")

    print(art_model.summary())

    return art_model

load_model()
