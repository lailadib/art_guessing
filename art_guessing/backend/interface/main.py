from backend.ml_logic.data import images_to_dataset
from backend.ml_logic.model import load_trained_model

def train():
    """
    - Create the training, validation and test dataset from the images

    """
    #Create train, validation and test dataset
    train_ds, val_ds, test_ds = images_to_dataset()

    print("✅ Data loaded and ready to use \n")

    #Train model using model.py

def predict():
    """
    Load the model if no model is already loaded
    Preprocess the new image
    Make prediction
    """
