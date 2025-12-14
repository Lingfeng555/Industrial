##
## segment.py
## Driver code to run the segmentation
##
## Diego Revilla
## Copyright (c) 2025 University of Deusto
##

import os
import cv2
from src.once_dataset import ONCEDataset
from ..classifier import SVMClassifier
from ..segmenter import CarSegmenter
from ..data_utils import extract_frame


class SegmentMode:
    def __init__(self, data_path, model_path, k = 2, 
                 low_threshold = 50, high_threshold = 150):
        self.data_path = data_path
        self.model_path = model_path
        self.classifier = SVMClassifier()
        self.segmenter = CarSegmenter(k=k, low_threshold=low_threshold, high_threshold=high_threshold)
    
    def run(self):
        self.classifier.load(self.model_path)
        
        splits = os.listdir(self.data_path)
        
        #carga cada split (train, val, test) en diferentes iteraciones
        for split in splits:
            dataset = ONCEDataset(
                data_path=self.data_path,
                split=split,
                data_type="both",
                level="frame",
                logger_name=f"ONCEDataset_{split}",
                show_logs=True
            )
            
            self._run_loop(dataset)
            cv2.destroyAllWindows()
    
    def _run_loop(self, dataset):        
        paused = False
        
        for idx in range(len(dataset)):
            if not paused:
                data = dataset[idx]
                
                #por cada frame, extrae la información de la cámara frontal
                cam_data = data['camera_data'].get("cam01")
                
                frame_bgr = extract_frame(cam_data)
                bboxes = cam_data.get('2D_bboxes', [])
                entities = cam_data.get('entities', [])
                
                #visualiza la información
                output = self.segmenter.visualize(frame_bgr, bboxes, entities, self.classifier)
                
                cv2.imshow("Car Segmentation", output)
            
            key = cv2.waitKey(30 if not paused else 1) & 0xFF
            
            #comandos de teclado
            if key == 27 or key == ord('q'):
                cv2.destroyAllWindows()
                return
            elif key == ord(' '):
                paused = not paused

