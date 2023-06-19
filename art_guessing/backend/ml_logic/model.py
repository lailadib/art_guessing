from tensorflow import data
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications.efficientnet import EfficientNetB2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
from params import *

def initialize_model(input_shape):
    """
    Initialize the Neural Network with transfer learning from EfficientNetB1
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
        # layers.RandomZoom(0.1),
        layers.RandomTranslation(0.2, 0.2),
        layers.RandomRotation(0.1)
    ])

    #Chain the petrained layers of EfficientNet with our own layers
    model = models.Sequential([
        layers.Input(shape = input_shape),
        augmentation,
        initialize_model(input_shape),
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
    Build the model from EfficientNetB1 and
    Compile the Neural Network
    """
    model = initialize_model((256,256,3))
    model = add_last_layers(model, (256,256,3))

    #Compile
    opt = optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy']
                 )

    print("✅ Model build and compiled")

    return model

def train_model(model, version: str, train_ds: data.Dataset, patience=4, validation_data=val_ds): # -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted model, history)
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
        "{}.h5".format(version),
        save_weights_only=True,
        monitor='val_accuracy',
        mode='auto',
        verbose=0,
        save_best_only=True
        )

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=200,
                        callbacks=[es, lr, mcp]
                        )

    return model, history

def load_trained_model():

    model = compile_model()
    model.load_weights(os.path.join(LOCAL_MODEL_PATH, "efficientnetb2_v2.h5"))
    print(model.summary())

    return model

def evaluate_model(model, test_ds: data.Dataset, verbose=0): # -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """
    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(test_ds, return_dict=True)

    loss = metrics['loss']
    accuracy = metrics['accuracy']

    print(f"✅ Model evaluated, Accuracy: {round(accuracy, 2)}")

    return metrics


load_trained_model()
