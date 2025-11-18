import streamlit as st
import sys
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage
import torch

sys.path.insert(0, './yolov12')

from yolov12.ultralytics import YOLO
from args import args
from src.once_dataset import ONCEDataset


# Load YOLOv12 model
segmentation_model = YOLO(model="yolov12m.pt")

# Load ONCE camera dataset
camera_dataset = ONCEDataset(
    data_path=args.data_path,
    split="val",
    data_type="camera",
    level="frame",
    logger_name=f"ONCEDataset_val",
    show_logs=False
)

# Initialize ToPILImage transform
to_pil = ToPILImage()

# Define inference function
def yolo_inference(image: torch.Tensor, confidence: float) -> np.ndarray:
    pil_img = to_pil(image * 255)
    results = segmentation_model.predict(source=pil_img, imgsz=640, conf=confidence)
    pred_entities = results[0].boxes.xyxy.cpu().numpy()
    pred_classes = results[0].boxes.cls.cpu().numpy()
    pred_conf = results[0].boxes.conf.cpu().numpy()

    image_rgb = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    for bbox, cls_id, conf in zip(pred_entities, pred_classes, pred_conf):
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        # Dibujar rectángulo
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Obtener nombre de clase
        class_name = segmentation_model.names[int(cls_id)]
        label = f"{class_name} {conf:.2f}"
        
        # Dibujar fondo para el texto
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(image_rgb, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), (0, 255, 0), -1)
        
        # Dibujar texto
        cv2.putText(image_rgb, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return pred_entities, image_rgb

# Streamlit App
st.title("Entities detection using camera data")
st.write("Visualization of detected entities on camera images using YOLOv12m model.")

selected_index = st.slider("Select Frame Index", 0, len(camera_dataset) - 1, 0)
selected_cam = st.selectbox("Select Camera", options=list(camera_dataset[0]["camera_data"].keys()), index=0)
selected_confidence = st.slider("Select Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

entities, bbbox_img = yolo_inference(camera_dataset[selected_index]["camera_data"][selected_cam]["image_tensor"], selected_confidence)

st.image(bbbox_img, caption=f"Frame Index: {selected_index}")
st.write(f"Detected {len(entities)} entities in the selected frame.")