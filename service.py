from __future__ import annotations

import json
import os
import typing as t
from pathlib import Path

import bentoml
import sys
from bentoml.validators import ContentType

import torch
import numpy as np
import cv2
from torchvision.transforms import ToPILImage

sys.path.append('./yolov12')
from yolov12.ultralytics import YOLO

@bentoml.service(resources={"gpu": 1})
class YoloService:

    model: YOLO
    to_pil: ToPILImage

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO("./yolov12m.pt").to(self.device)
        self.to_pil = ToPILImage()

    @bentoml.api
    def yolo_inference(self, image: torch.Tensor, confidence: float)-> dict[str, t.Any] :
        pil_img = self.to_pil(image)
        results = self.model.predict(source=pil_img, imgsz=640, conf=confidence)
        pred_entities = results[0].boxes.xyxy.cpu().numpy()
        pred_classes = results[0].boxes.cls.cpu().numpy()
        pred_conf = results[0].boxes.conf.cpu().numpy()

        image_rgb = (image.permute(1, 2, 0).contiguous() .cpu().numpy() * 255).astype(np.uint8)
        
        for bbox, cls_id, conf in zip(pred_entities, pred_classes, pred_conf):
            x1, y1, x2, y2 = map(int, bbox[:4])
            
            # Dibujar rectángulo
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Obtener nombre de clase
            class_name = self.model.names[int(cls_id)]
            label = f"{class_name} {conf:.2f}"
            
            # Dibujar fondo para el texto
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image_rgb, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), (0, 255, 0), -1)
            
            # Dibujar texto
            cv2.putText(image_rgb, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return {"pred_entities": pred_entities.tolist(), "image_rgb": image_rgb.tolist()}