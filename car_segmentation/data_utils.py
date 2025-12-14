##
## data_utils.py
## utility functions for data extraction and processing
##
## Diego Revilla (with the help of ChatGPT)
## Copyright (c) 2025 University of Deusto
##

import cv2
import numpy as np
import random
from collections import Counter, defaultdict
from tqdm import tqdm
from typing import List, Tuple, Optional


def extract_frame(cam_data):
    image_tensor = cam_data.get('image_tensor')
    frame = np.asarray(image_tensor)
    frame = frame.transpose(1, 2, 0)
    frame = (frame * 255).astype(np.uint8)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame_bgr


def map_label(label: str):
    # No detectamos peatones ni ciclistas por falta de datos en estas clases
    if label in ['Pedestrian', 'Cyclist']:
        return 'Other'
    return label


def collect_samples(dataset, augment_minority = True, max_box_ratio = 0.3, sample_ratio = 1.0):

    minority_classes = ['Bus', 'Truck']
    
    total = len(dataset)
    samples_by_class = defaultdict(list)
    
    #collecionamos todas las muestras de un video en concreto, y obtenemos las entidades de cada frame
    for idx in tqdm(range(total), desc="Collecting samples"):
        data = dataset[idx]
        cam_data = data['camera_data'].get("cam01")
        
        frame_bgr = extract_frame(cam_data)
        bboxes = cam_data.get('2D_bboxes', [])
        entities = cam_data.get('entities', [])
        
        h, w = frame_bgr.shape[:2]
        frame_area = h * w
        
        for bbox, label in zip(bboxes, entities):            
            x1, y1, x2, y2 = map(int, bbox[:4])
            
            # Clamp coordinates to frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # Skip invalid bounding boxes
            if x1 >= x2 or y1 >= y2:
                continue
            
            box_area = (x2 - x1) * (y2 - y1)

            # PRE-PROCESADO. Si la caja es más de 0.3 de la imagen total, la descartamos
            if box_area > max_box_ratio * frame_area:
                continue
            
            roi = frame_bgr[y1:y2, x1:x2].copy()
            
            # Skip empty ROIs
            if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
                continue
                
            mapped_label = map_label(label)
            samples_by_class[mapped_label].append(roi)

    images = []
    labels = []
    
    #creamos el dataset final a partir de las imagenenes etiquetadas y también de las 
    # instancias flipped (para las clases sub-representadas)
    for class_name, class_samples in samples_by_class.items():

        for sample in class_samples:
            images.append(sample)
            labels.append(class_name)

        if class_name in minority_classes:
            for sample in class_samples:
                flipped = cv2.flip(sample, 1)
                images.append(flipped)
                labels.append(class_name)
    
    return images, labels
