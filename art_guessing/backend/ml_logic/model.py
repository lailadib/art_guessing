from tensorflow import data
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications.efficientnet import EfficientNetB2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from os.path import join
from params import *
import numpy as np

def initialize_model(input_shape):
    """
    Initialize the Neural Network with transfer learning from EfficientNetB2
    """
    model = EfficientNetB2(weights='imagenet', include_top=False, input_shape=input_shape)
    #Set the first layers to be untrainable
    model.trainable = False

    print("✅ Model initialized")

    return model

def add_last_layers(model, input_shape):
    """
    Add the last layers of the model
    """
    #Data augmentation
    augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(0.2, 0.2),
        layers.RandomRotation(0.1)
    ])

    #Chain the petrained layers of EfficientNetB2 with our own layers
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
    Build the model from EfficientNetB2 and
    Compile the Neural Network
    """
    #Build the model
    model = add_last_layers(initialize_model(INPUT_SHAPE), INPUT_SHAPE)

    #Compile the model
    opt = optimizers.Adam(learning_rate=0.001)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy']
                 )

    print("✅ Model build and compiled")

    return model

def train_model(model, model_name: str, train_ds, val_ds):
    """
    Fit the model
    Saved a .h5 file with the trained weights with version name
    Return a tuple (fitted model, history)
    """
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

    mcp = ModelCheckpoint(
        "{}.h5".format(model_name),
        save_weights_only=True,
        monitor='val_accuracy',
        mode='auto',
        verbose=0,
        save_best_only=True,
        filepath=os.path.join(LOCAL_MODEL_PATH, "{}.h5".format(model_name))
        )

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=200,
                        callbacks=[es, lr, mcp]
                        )

    return model, history

def load_trained_model(model_name=None):
    """
    Load the model with pre-trained weights
    """
    model = compile_model()

    if model_name == None:
        filename = STD_MODEL_NAME #load standard model if not given anothe one
    else:
        filename = model_name

    print("Loading trained model... \n")

    try:
        model.load_weights(os.path.join(LOCAL_MODEL_PATH, filename))
    except:
        print(f"This file: {os.path.join(LOCAL_MODEL_PATH, filename)} does not exist!")
        return None

    print(model.summary())
    return model

def evaluate_model(model, test_ds: data.Dataset, verbose=0):
    """
    Evaluate trained model performance on the dataset
    """
    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(test_ds, return_dict=True)

    accuracy = metrics['accuracy']

    print(f"✅ Model evaluated, Accuracy: {round(accuracy, 2)}")

    return metrics

def predict(model, new_image):
    """
    Make predictions from a preprocessed image
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
