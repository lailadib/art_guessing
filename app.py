import streamlit as st
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np

URL = "https://en.wikipedia.org/w/api.php"
PARAMS = {
        "action": "opensearch",
        "namespace": "0",
        "search": 'some_style', # to exchange before query
        "limit": "5",
        "format": "json"}
CLASSIFIED_STYLE = 'Abstract'
API_RESPONSE = None
styles = ('Expressionism',
                 'Impressionism',
                 'Abstract',
                 'Surrealism') #add more here

st.title('Guess the arts')
img = Image.open('images/IMG_0071.JPG')
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
                API_RESPONSE = True # needs better solution
                if guessed_style == CLASSIFIED_STYLE:
                    st.subheader('Well done! you are the king of the arts!')
                    st.balloons()
                else:
                    st.subheader('Are you kidding me? Go home and study arts!')
                    st.snow()
                    PARAMS['search'] = guessed_style
                    res = requests.get(url=URL, params=PARAMS).json()
                    st.write(res[3][0])
            else:
                st.subheader('Please choose your pic first!')
with pred_col2:
    if uploaded_file != None:
        st.image(uploaded_file, width=350)

with st.expander('Look into the kitchen!'): # show plots or other usefull info behind the wall
    if API_RESPONSE != None:
        arr = np.random.normal(1, 1, size=100)
        fig, axs = plt.subplots(1, 2)
        axs[0].hist(arr, bins=20)
        axs[1].hist(arr, bins=20)
        st.pyplot(fig)
