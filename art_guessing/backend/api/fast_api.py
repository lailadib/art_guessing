from fastapi import FastAPI, UploadFile, File
from PIL import Image
import os
import numpy as np
import cv2

from backend.ml_logic.registry import load_model


#load_dotenv()
api_img_endpoint = os.getenv('API_IMAGE_ENDPOINT')
port = os.getenv('PORT')

app = FastAPI()

# preload the model
app.state.model = load_model()

if app.state.model != None:
    print('Model loaded')

@app.get("/")
def index():
    return {"status": "ok"}


@app.post('/upload_image')
def receive_image(file: UploadFile):
    ### Receiving and decoding the image

    img = Image.open(file.file) # it is tested: can be opened

    ### Do your image classification stuff here....

    predicted_style = 'surrealism' # call pred func hier, provide image and model to use

    ### print(f'classified as: {preddicted_style}')
    return {'style': predicted_style}
