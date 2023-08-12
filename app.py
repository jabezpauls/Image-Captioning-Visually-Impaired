import streamlit as st
import cv2
import os
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

# Load the image captioning model and components
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Function to generate image captions
def generate_captions(image_path):
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

    images = [i_image]
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    caption = preds[0].strip()

    return caption

def main():
    st.markdown(
        """
        <style>
        body {
            background-color: lightblue;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Image Captioning using Camera Feed")

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.write("Error: Could not access the camera")
        return
    
    capture_button = st.button("Capture and Predict")
    video_placeholder = st.empty()
    captured_image = None

    while True:
        ret, frame = cap.read()
        
        if not ret:
            st.write("Error: Could not read frame")
            break
        
        video_placeholder.image(frame, channels="BGR", use_column_width=True)
        
        if capture_button:
            captured_image = frame
            capture_button = False
    
        if captured_image is not None:
            st.image(captured_image, channels="BGR", use_column_width=True)
            
            image_path = "captured_image.jpg"
            cv2.imwrite(image_path, captured_image)
            caption = generate_captions(image_path)
            st.write("Predicted Caption:", caption)
            captured_image = None

    cap.release()

if __name__ == "__main__":
    main()
