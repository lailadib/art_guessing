from fastapi import FastAPI, UploadFile
from backend.ml_logic.model import load_trained_model, predict
from backend.ml_logic.data import preprocess_new_image

app = FastAPI()

# preload the stadard model from local directory given by LOCAL_MODEL_PATH
app.state.model = load_trained_model()

# creates wrapper
if app.state.model != None:
    print('✅ Model loaded')

@app.get("/")
def index():
    return {"status": "art guessing is greeting you"}


@app.post('/upload_image')
def receive_image(file: UploadFile):
    """ Retrives image file from the frontend.
    Returns prediction of the artwork given on image
    as list of two dictionaries like:
    0. {'style': 'some_style', 'proba': 'some_proba'}  # this is the best guess
    1. {'some_style1': 'proba1', 'some_style2': 'proba2', ..., 'some_style10':  'proba10'} # this are all probablitys
    file: UploadFile -- image file
    """

    # Translate the image file to tensor first.
    # Image coming from the frontend is expected to be:
    # squared, downsized and padded (no border or part of frame should be seen)
    # otherwise this can be done by preprocess_new_image on request
    img_tens = preprocess_new_image(file.file)
    predicted = predict(app.state.model, img_tens) # -> list of 2 dicts

    return predicted
