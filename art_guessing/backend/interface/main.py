from backend.ml_logic.data import images_to_dataset
from backend.ml_logic.model import train_model
from backend.ml_logic.model import evaluate_model
from backend.ml_logic.model import compile_model
from backend.ml_logic.model import load_trained_model
import datetime
from params import *


def train(new_model_name):
    """ trains and saves a new model """

    # loads data set from lacal dir ./raw_images
    train_ds, val_ds, test_ds = images_to_dataset()

    # creats basic model
    basic_model = compile_model()

    # trains model on data set and saves a new one with the given name in local dir ./models
    trained_model, history = train_model(basic_model, new_model_name, train_ds, val_ds)

    return (trained_model, history)

def evaluate(model_name):
    """ evaluates a model  """

    #loads model with a given name from the local folder ./models
    model = load_trained_model(model_name)
    if model == None:
        print('❌ No model created')
        return None

    # creates test_data
    train_ds, val_ds, test_ds = images_to_dataset()

    # evaluates model and prints out the accuracy
    evaluate_model(model, test_ds, verbose=1)


if __name__ == '__main__':

    # create a new name for the model
    # new_model_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # train the new model
    # trained_model, history = train(model_name) # new model is saved to ./models

    # process the images (make tensors)
    # train_ds, val_ds, test_ds = images_to_dataset()

    # evaluate model (metric will be printed)
    # evaluate(model_name, test_ds, verbose=1)

    # ------------------------------------------------
    # just for testing of loading and evaluating of existing model
    model_name = STD_MODEL_NAME
    train_ds, val_ds, test_ds = images_to_dataset()
    evaluate(model_name, test_ds, verbose=1)
    # ------------------------------------------------
