from fastapi import FastAPI, UploadFile, File
import os
import numpy as np
import cv2

from ml_logic.registry import load_model


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
async def receive_image(img: UploadFile=File(...)):
    ### Receiving and decoding the image
    contents = await img.read()

    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray

    ### Do your image classification stuff here....

    predicted_style = 'surrealism' # call pred func hier, provide image and model to use

    ### print(f'classified as: {preddicted_style}')
    return {'style': predicted_style}
