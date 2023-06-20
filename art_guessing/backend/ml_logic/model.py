from tensorflow import data
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications.efficientnet import EfficientNetB2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from os.path import join
from params import *
import numpy as np

def initialize_model(input_shape):
    """Initialize the Neural Network with transfer learning from EfficientNetB2
    ----------
    Arguments:
    input_shape -- Input shape given to the model i.e (256,256,3)
    -----------
    Returns a model that doesn't include top layers and has non-trainable weights
    """
    model = EfficientNetB2(weights='imagenet', include_top=False, input_shape=input_shape)
    #Set the first layers to be non-trainable
    model.trainable = False

    print("✅ Model initialized")

    return model

def add_last_layers(model, input_shape):
    """
    Add the customized last layers to the model
    ----------
    Arguments:
    model -- model initialized with EfficientNetB2
    input_shape -- Input shape given to the model i.e (256,256,3)
    -----------
    Returns a model with its complete architecture
    """
    #Data augmentation
    augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(0.2, 0.2),
        layers.RandomRotation(0.1)
    ])

    #Chain the pre-trained layers of EfficientNetB2 with our own layers
    base_model = initialize_model(input_shape)

    model = models.Sequential([
        layers.Input(shape = input_shape),
        augmentation,
        base_model,
        layers.Flatten(),
        layers.Dense(300, activation='gelu'), #'gelu'
        layers.Dropout(0.25),
        layers.Dense(150, activation='gelu'),
        layers.Dropout(0.25),
        layers.Dense(10, activation='softmax')
    ])

    print("✅ Last layers initialized")

    return model

def compile_model():
    """
    Build the model from EfficientNetB2 and custom layers, then
    Compile the Neural Network
    ----------
    Returns a compiled model
    """
    #Build model
    model = add_last_layers(initialize_model(INPUT_SHAPE), INPUT_SHAPE)

    #Compile model
    opt = optimizers.Adam(learning_rate=0.001)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy']
                 )

    print("✅ Model build and compiled")

    return model

def train_model(model, model_name: str, train_ds, val_ds):
    """
    Fit the model on the training dataset.
    Saved a .h5 file with the trained weights with model_name
    ----------
    Return a tuple (fitted model, history)
    """
    #Set the callbacks
    es = EarlyStopping(
        monitor='val_accuracy',
        mode='auto',
        patience=4,
        verbose=1,
        restore_best_weights=True
        )

    lr = ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.05,
        patience=2,
        verbose=1,
        min_lr=0
        )

    #Save the weights of the model as a .h5 file inside /models folder
    mcp = ModelCheckpoint(
        save_weights_only=True,
        monitor='val_accuracy',
        mode='auto',
        verbose=0,
        save_best_only=True,
        filepath=os.path.join(LOCAL_MODEL_PATH, "{}.h5".format(model_name))
        )

    #Fit the model on training dataset
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=200,
                        callbacks=[es, lr, mcp]
                        )

    return model, history

def load_trained_model(model_name=None):
    """Build the model and load the pre-trained weights from .h5 file
    ----------
    model_name -- '.h5' filename of the pre-trained weights we want to load
    ----------
    Returns a model with its complete architecture
    """
    model = compile_model()

    if model_name == None:
        filename = STD_MODEL_NAME #Load standard model if not given another one
    else:
        filename = model_name

    print("Loading trained model... \n")

    try:
        model.load_weights(os.path.join(LOCAL_MODEL_PATH, filename))
        print(f"✅ Model {os.path.join(LOCAL_MODEL_PATH, filename)} has been loaded")
    except:
        print(f"❌ This file: {os.path.join(LOCAL_MODEL_PATH, filename)} does not exist!")
        return None

    print(model.summary())

    return model

def evaluate_model(model, test_ds: data.Dataset, verbose=0):
    """Evaluate trained model performance on the dataset
    ----------
    Arguments:
    model -- trained model
    test_ds -- test dataset used to evaluate the model. Could be a tensor or tf.data.Dataset
    ----------
    Returns a dictionnary of metrics
    """
    if model is None:
        print(f"❌ No model to evaluate")
        return None

    metrics = model.evaluate(test_ds, return_dict=True)

    accuracy = metrics['accuracy']

    print(f"✅ Model evaluated, Accuracy: {round(accuracy, 2)}")

    return metrics

def predict(model, new_image):
    """Make predictions from a preprocessed image
    ----------
    Arguments:
    model -- trained model
    new_image -- image as a tensor of size (1,256,256,3)
    ----------
    Returns a tuple of dictionaries :
    - first_prediction contains the highest probability with its category given by the model
    - predictions contains the probabilities for each category
    """
    y_pred = model.predict(new_image).tolist()[0]

    predictions = {CLASS_NAMES[i]: np.round(y_pred[i], 2) for i in range(len(CLASS_NAMES))}

    if max(predictions.values()) < 0.2:
        first_prediction = {'style': None, 'probability': max(predictions.values())}
        return (first_prediction, predictions)

    first_prediction = {'style': max(predictions, key=predictions.get), 'probability': max(predictions.values())}

    return (first_prediction, predictions)
