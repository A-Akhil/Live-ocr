import streamlit as st
import google.generativeai as genai
import google.ai.generativelanguage as glm
from PIL import Image
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import numpy as np
import threading
import time

API_KEY = 'AIzaSyBAbD0NIH0h8xMHAGsVjFvJlOQFT7oxl8E'
genai.configure(api_key=API_KEY)

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
        self.ocr_lock = threading.Lock()
        self.running_ocr = False
        self.last_ocr_time = time.time()  # Initialize last OCR time

    def recv(self, frame):
        # Debug: Frame received
        self.frame = frame.to_ndarray(format="bgr24")
        
        # Check if 2 seconds have passed since the last OCR run
        current_time = time.time()
        if not self.running_ocr and (current_time - self.last_ocr_time) >= 2.0:  # Changed to 2 seconds
            self.last_ocr_time = current_time  # Update the last OCR run time
            threading.Thread(target=self.run_ocr).start()

        return frame

    def run_ocr(self):
        self.ocr_lock.acquire()
        self.running_ocr = True
        try:
            if self.frame is not None:
                # Process OCR in background thread, but update result in main thread using session_state
                image = Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
                img_bytes = cv2.imencode('.jpg', self.frame)[1].tobytes()

                # Send the image to the model for OCR
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(
                    glm.Content(
                        parts=[
                            glm.Part(text="OCR this handwritten text and provide the result."),
                            glm.Part(
                                inline_data=glm.Blob(
                                    mime_type='image/jpeg',
                                    data=img_bytes
                                )
                            ),
                        ]
                    ),
                    stream=True
                )

                response.resolve()

                # Update the session_state with the OCR result
                st.session_state.ocr_result = response.text

        finally:
            self.running_ocr = False
            self.ocr_lock.release()


# Initialize the webrtc streamer for live video capture
webrtc_ctx = webrtc_streamer(
    key="live-ocr", 
    video_processor_factory=OCRVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

# Display the real-time OCR result in the main thread
st.subheader("Detected Text")
st.write(st.session_state.ocr_result or "Writing detected text will appear here...")
