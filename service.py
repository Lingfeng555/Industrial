from __future__ import annotations

import json
import os
import typing as t
from pathlib import Path

import bentoml
import sys
import base64  # Added import
from bentoml.validators import ContentType

import torch
import numpy as np
import cv2
from torchvision.transforms import ToPILImage
import time
sys.path.append('./yolov12')
from yolov12.ultralytics import YOLO

@bentoml.service(resources={"gpu": 1})
class YoloService:

    model: YOLO
    to_pil: ToPILImage

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO("./yolov12m.pt").to(self.device)
        # self.to_pil = ToPILImage() # No longer needed

    @bentoml.api
    def yolo_inference(self, image_b64: str, confidence: float)-> dict[str, t.Any] :
        # 1. Decode Base64 Input to BGR Image
        img_data = base64.b64decode(image_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
       
        # 2. Run Inference (YOLO accepts BGR numpy arrays directly)
        results = self.model.predict(source=image_bgr, imgsz=640, conf=confidence)
        
        start = time.time()
        pred_entities = results[0].boxes.xyxy.cpu().numpy()
        pred_classes = results[0].boxes.cls.cpu().numpy()
        pred_conf = results[0].boxes.conf.cpu().numpy()

        # 3. Draw on the BGR image
        for bbox, cls_id, conf in zip(pred_entities, pred_classes, pred_conf):
            x1, y1, x2, y2 = map(int, bbox[:4])
            
            # Dibujar rectángulo
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Obtener nombre de clasef
            class_name = self.model.names[int(cls_id)]
            label = f"{class_name} {conf:.2f}"
            
            # Dibujar fondo para el texto
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image_bgr, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), (0, 255, 0), -1)
            
            # Dibujar texto
            cv2.putText(image_bgr, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Encode result BGR image to JPEG Base64
        _, buffer = cv2.imencode('.jpg', image_bgr)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        print(f"YOLO inference time: {time.time() - start:.3f}s")
        return {"pred_entities": pred_entities.tolist(), "image_base64": img_base64}


@bentoml.service
class ImageServerService:
    
    def __init__(self):
        self.output_dir = Path("inference_outputs")
        self.output_dir.mkdir(exist_ok=True)
    
    @bentoml.api
    def get_image(self, filename: str) -> dict[str, t.Any]:
        """Serve an image from inference_outputs directory as base64"""
        file_path = self.output_dir / filename
        
        if not file_path.exists():
            return {"error": f"Image {filename} not found"}
        
        try:
            with open(file_path, "rb") as f:
                image_data = f.read()
            img_base64 = base64.b64encode(image_data).decode('utf-8')
            return {"filename": filename, "image_base64": img_base64}
        except Exception as e:
            return {"error": str(e)}
    
    @bentoml.api
    def list_images(self) -> dict[str, list[str]]:
        """List all images in inference_outputs directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        images = [
            f.name for f in self.output_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        return {"images": sorted(images)}