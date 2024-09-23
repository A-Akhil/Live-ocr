import streamlit as st
from groq import Groq
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import time
from PIL import Image
import numpy as np
import base64

API_KEY = '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' 
client = Groq(api_key=API_KEY)

st.set_page_config(page_title="Live Handwriting OCR", 
                   page_icon="📸",
                   layout="centered",
                   initial_sidebar_state='collapsed')

st.header("Real-Time Handwriting OCR")

# Initialize the session state for OCR result if not already set
if "ocr_result" not in st.session_state:
    st.session_state.ocr_result = ""

class OCRVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame = None
        self.running_ocr = False
        self.last_ocr_time = time.time()  # Initialize last OCR time

    def recv(self, frame):
        self.frame = frame.to_ndarray(format="bgr24")
        
        # Check if 3 seconds have passed since the last OCR run
        current_time = time.time()
        if not self.running_ocr and (current_time - self.last_ocr_time) >= 3.0:
            self.last_ocr_time = current_time
            self.run_ocr()  # Call run_ocr directly

        return frame

    def run_ocr(self):
        if self.frame is not None:
            self.running_ocr = True
            try:
                img_bytes = cv2.imencode('.jpg', self.frame)[1].tobytes()

                # Send the image to the Groq model for OCR
                response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "OCR the image and tell me what is the text in image alone"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode()
                                    },
                                },
                            ],
                        }
                    ],
                    model="llava-v1.5-7b-4096-preview",
                )

                # Update the session_state with the OCR result
                st.session_state.ocr_result = response.choices[0].message.content

                # Display the detected text in real-time
                st.subheader("Detected Text")
                st.write(st.session_state.ocr_result or "Writing detected text will appear here...")

            finally:
                self.running_ocr = False

# Initialize the webrtc streamer for live video capture
webrtc_ctx = webrtc_streamer(
    key="live-ocr", 
    video_processor_factory=OCRVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

# Display the real-time OCR result (initial state)
st.subheader("Detected Text")
st.write(st.session_state.ocr_result or "Writing detected text will appear here...")
