from fastapi import FastAPI, UploadFile, File
from PIL import Image
import os
import numpy as np
import cv2
from backend.ml_logic.model import load_trained_model, predict
from backend.ml_logic.data import preprocess_new_image

app = FastAPI()

# preload the model
app.state.model = load_trained_model()

if app.state.model != None:
    print('✅ Model loaded')

@app.get("/")
def index():
    return {"status": "art guessing is greeting you"}


@app.post('/upload_image')
def receive_image(file: UploadFile):
    #Test 1
    ### Receiving and decoding the image (Option 1):
    #img = Image.open(file.file) # it is tested: can be opened
    ### Do your image classification stuff here....
    #predicted = predict(app.state.model, img) #sending jpeg

    #Test 2 (prefer if works)
    ### Receiving an image file and sending it unchagenged to the predict function (Option 2):
    ### translate to tensor first. image coming from the front end already square, downsized and padded
    img_tens = preprocess_new_image(file.file)
    predicted = predict(app.state.model, img_tens)

    # 1. predicted_style {'style': 'some_style', 'proba': 'some_proba'}
    # 2. complete table {'some_style1': 'proba1, 'some_style2': 'proba2, ... till the end}

    return predicted
    #return {'style': predicted_style}
