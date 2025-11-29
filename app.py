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

from src.once_dataset import ONCEDataset
from car_simulator import Car

available_cars = ["000027","000028", "000112", "000201"]
cars = {}
for id in available_cars:
    cars[id] = Car(car_id=id, data_path=args.data_path)

cams = cars[id].cams

# Define inference function
def yolo_api_inference(image: torch.Tensor, confidence: float) -> np.ndarray:
    client = bentoml.SyncHTTPClient('http://localhost:3000')
    print("Sending inference request to YOLO service...")
    if client.is_ready():
        response = client.yolo_inference(image=image, confidence=confidence)
        entities = response["pred_entities"]
        bbbox_img = np.array(response["image_rgb"])
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
@st.fragment(run_every="1500ms")
def show_car_cam():
    start = time.time()
    car = cars[selected_car]
    lidar_data, images = car.get_info()
    image = images[selected_cam]

    print("Inference started")
    entities, bbbox_img = yolo_api_inference(image, selected_confidence)
    print("Inference completed")

    st.image(bbbox_img, caption=f"From car {selected_car}") 
    st.write(f"Detected {len(entities)} entities in the selected frame.")
    st.write(f"Frame time: {time.time() - start:.3f}s")

    st.session_state.cam_busy = False

show_car_cam()