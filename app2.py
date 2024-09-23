import streamlit as st
import google.generativeai as genai
import google.ai.generativelanguage as glm
from PIL import Image

API_KEY = 'AIzaSyBAbD0NIH0h8xMHAGsVjFvJlOQFT7oxl8E'
genai.configure(api_key=API_KEY)

st.set_page_config(page_title="Handwriting OCR", 
                   page_icon="📸",
                   layout="centered",
                   initial_sidebar_state='collapsed')

st.header("Handwriting OCR From Image")

# Use the camera input instead of file uploader
captured_image = st.camera_input("Capture Image")

if captured_image is not None:
    image = Image.open(captured_image)

    st.image(image, caption='Captured Image', use_column_width=True)
    bytes_data = captured_image.getvalue()

    generate = st.button("OCR!")

    if generate:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            glm.Content(
                parts=[
                    glm.Part(text="OCR and give the text from this picture, it's handwritten so make sure you adapt to its style."),
                    glm.Part(
                        inline_data=glm.Blob(
                            mime_type='image/jpeg',
                            data=bytes_data
                        )
                    ),
                ],
            ),
            stream=True
        )

        response.resolve()

        st.write(response.text)
