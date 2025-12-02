from unittest import result
import streamlit as st
import sys
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage
import torch
import requests
from args import args
from io import BytesIO
import bentoml
import time
import base64 # Added import

from src.once_dataset import ONCEDataset
from car_simulator import Car

available_cars = ["000027","000028", "000112", "000201"]
cars = {}
for id in available_cars:
    cars[id] = Car(car_id=id, data_path="/home/none3075/Desktop/asignaturas/industria/proyecto/Industrial/data")

cams = cars[id].cams
client = bentoml.SyncHTTPClient('http://localhost:3000')

# Define inference function
def yolo_api_inference(image: torch.Tensor, confidence: float) -> np.ndarray:
    print("Sending inference request to YOLO service...")
    response = client.yolo_inference(image=image, confidence=confidence)
    entities = response["pred_entities"]
    
    # Decode Base64 string back to image
    img_data = base64.b64decode(response["image_base64"])
    bbbox_img = np.array(Image.open(BytesIO(img_data)))
    
    print(bbbox_img.shape)
    return entities, bbbox_img

# Streamlit App
if "cam_busy" not in st.session_state:
    st.session_state.cam_busy = False

st.title("Driver Monitor")
st.write("Visualize the driving experience of the agents")

selected_car = st.selectbox("Select Car", options=available_cars, index=0)
selected_cam = st.selectbox("Select Camera", options=cars[selected_car].cams, index=0)
selected_confidence = st.slider("Select Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

#Read in "real time"
@st.fragment()  # Runs at 20 Hz
@st.fragment(run_every="150ms")
def show_car_cam():
    start = time.time()
    car = cars[selected_car]
    lidar_data, images = car.get_info()
    image = images[selected_cam]

    print("Inference started")
    start_inference = time.time()
    entities, bbbox_img = yolo_api_inference(image, selected_confidence)
    print("Inference completed")
    print(f"Inference time: {time.time() - start_inference:.3f}s")

    st.image(bbbox_img, caption=f"From car {selected_car}") 
    st.write(f"Detected {len(entities)} entities in the selected frame.")
    st.write(f"Frame time: {time.time() - start:.3f}s")

    st.session_state.cam_busy = False

show_car_cam()