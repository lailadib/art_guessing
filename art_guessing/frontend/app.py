import streamlit as st
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
import os

#load_dotenv()
wiki_url = "https://en.wikipedia.org/w/api.php"
api_url = os.getenv('API_URL')
api_img_endpoint = os.getenv('API_IMAGE_ENDPOINT')

# these are params for the wiki request
PARAMS = {
        "action": "opensearch",
        "namespace": "0",
        "search": 'some_style', # to exchange before query
        "limit": "5",
        "format": "json"}

api_response = None

#styles = ('Expressionism',
#                 'Impressionism',
#                 'Abstract',
#                 'Surrealism') #add more here

styles = ('art nouveau',
            'baroque',
            'expressionism',
            'impressionism',
            'post-impressionism',
            'realism',
            'renaissance',
            'romanticism',
            'surrealism',
            'ukiyo-e')

st.title('Guess the arts')
img = Image.open('frontend/images/IMG_0071.JPG')
st.image(img, width=100)

#pred, tech,  = st.tabs(['Test your knowlede', 'Tech. backgroud'])
pred_col1, pred_col2 = st.columns(2)
st.divider()
#with pred:
with pred_col1:
    st.subheader('Choose your pic!')
    uploaded_file = st.file_uploader("", accept_multiple_files=False)
    st.divider()
    st.subheader('Make your guess!')
    guessed_style = st.radio('', styles)
    st.divider()
    st.subheader('Test your knowlede!')
    if st.button('Click me!'):
        with st.spinner('Be patient ... we are asking ChatGPT'):
            st.divider()
            if uploaded_file != None:
                # transfer image to bytes
                img_bytes = uploaded_file.getvalue() #encoding to binary
                # make request to the API
                api_response = requests.post(api_url + api_img_endpoint,
                                             files={'img': img_bytes})
                # for testing only
                #api_response = requests.get(api_url + api_img_endpoint) # api_img_endpoint)

                classified_style = api_response.json()['style']

                if api_response!= None and api_response.status_code == 200:
                    #classified_style = classified_style
                    if guessed_style == classified_style:
                        st.subheader(f'Well done! It is {classified_style}! You are the king of the arts!')
                        st.balloons()
                    else:
                        st.subheader(f'Are you kidding me? Go home and study arts!')
                        st.snow()
                        PARAMS['search'] = guessed_style
                        res = requests.get(url=wiki_url, params=PARAMS).json()
                        st.write(res[3][0])
                else:
                    st.write(f'api responded with: {api_response}')
            else:
                st.subheader('Please choose your pic first!')
with pred_col2:
    if uploaded_file != None:
        st.image(uploaded_file, width=350)

with st.expander('Look into the kitchen!'): # show plots or other usefull info behind the wall
    if api_response!= None and api_response.status_code == 200:
        arr = np.random.normal(1, 1, size=100) #some stuff to look at until real data is there
        fig, axs = plt.subplots(1, 2)
        axs[0].hist(arr, bins=20)
        axs[1].hist(arr, bins=20)
        st.pyplot(fig)
