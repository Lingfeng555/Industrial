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
image_server_client = bentoml.SyncHTTPClient('http://localhost:3001')

# Define inference function
def yolo_api_inference(image: torch.Tensor, confidence: float) -> np.ndarray:
    img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    _, buffer = cv2.imencode('.jpg', img_bgr)
    img_base64_input = base64.b64encode(buffer).decode('utf-8')

    print("Sending inference request to YOLO service...")
    
    response = client.yolo_inference(image_b64=img_base64_input, confidence=confidence)
    
    entities = response["pred_entities"]
    
    img_data = base64.b64decode(response["image_base64"])
    bbbox_img = np.array(Image.open(BytesIO(img_data)))
    
    print(bbbox_img.shape)
    return entities, bbbox_img

# Streamlit App
if "cam_busy" not in st.session_state:
    st.session_state.cam_busy = False

if "current_image_index" not in st.session_state:
    st.session_state.current_image_index = 0

st.title("Driver Monitor")
st.write("Visualize the driving experience of the agents")

# Create tabs for different views
tab1, tab2 = st.tabs(["Live Camera", "Prediction Images"])

with tab1:
    selected_car = st.selectbox("Select Car", options=available_cars, index=0)
    selected_cam = st.selectbox("Select Camera", options=cars[selected_car].cams, index=0)
    selected_confidence = st.slider("Select Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

    #Read in "real time"
    @st.fragment(run_every=0.1)
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

with tab2:
    st.subheader("Saved Inference Frames")
    try:
        response = image_server_client.list_images()
        images_list = response.get("images", [])
        
        if not images_list:
            st.info("No images found in inference_outputs directory")
        else:
            st.write(f"Found {len(images_list)} saved frames")
            
            
            
            # Image display with auto-play
            @st.fragment(run_every=0.5)
            def show_image_sequence():
                if not images_list:
                    return
                
                st.session_state.current_image_index = (st.session_state.current_image_index + 1) % len(images_list)
                
                # Ensure index is within bounds
                if st.session_state.current_image_index >= len(images_list):
                    st.session_state.current_image_index = 0
                
                current_image = images_list[st.session_state.current_image_index]
                
                # Fetch and display the current image
                img_response = image_server_client.get_image(filename=current_image)
                
                if "error" in img_response:
                    st.error(f"Error loading image: {img_response['error']}")
                else:
                    img_data = base64.b64decode(img_response["image_base64"])
                    img = Image.open(BytesIO(img_data))
                    st.image(img, caption=f"{current_image} ({st.session_state.current_image_index + 1}/{len(images_list)})", width='stretch')
            
            show_image_sequence()
            
            new_index = st.slider(
                "Frame", 
                min_value=0, 
                max_value=len(images_list) - 1, 
                value=st.session_state.current_image_index,
                key="frame_slider"
            )
            if new_index != st.session_state.current_image_index:
                st.session_state.current_image_index = new_index
                    
    except Exception as e:
        st.error(f"Could not connect to Image Server Service: {str(e)}")
        st.info("Make sure the ImageServerService is running on port 3001")