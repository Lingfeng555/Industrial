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
from src.once_dataset import ONCEDataset
from io import BytesIO
import bentoml

# Load ONCE camera dataset
camera_dataset = ONCEDataset(
    data_path=args.data_path,
    split="val",
    data_type="camera",
    level="frame",
    logger_name=f"ONCEDataset_val",
    show_logs=False
)

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
st.title("Entities detection using camera data")
st.write("Visualization of detected entities on camera images using YOLOv12m model.")

selected_index = st.slider("Select Frame Index", 0, len(camera_dataset) - 1, 0)
selected_cam = st.selectbox("Select Camera", options=list(camera_dataset[0]["camera_data"].keys()), index=0)
selected_confidence = st.slider("Select Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

entities, bbbox_img = yolo_api_inference(camera_dataset[selected_index]["camera_data"][selected_cam]["image_tensor"], selected_confidence)

st.image(bbbox_img, caption=f"Frame Index: {selected_index}")
st.write(f"Detected {len(entities)} entities in the selected frame.")