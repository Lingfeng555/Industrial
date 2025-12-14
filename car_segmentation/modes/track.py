##
## track.py
## Driver code to track cars using KCF tracker
##
## Diego Revilla (with the help of ChatGPT)
## Copyright (c) 2025 University of Deusto
##

import os
import cv2
from src.once_dataset import ONCEDataset
from ..tracker import KCFTracker
from ..data_utils import extract_frame

class TrackMode:
    def __init__(self, data_path):
        self.data_path = data_path
    
    def run(self):        
        splits = os.listdir(self.data_path)
        
        #load one dataset at a time
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

    def restart_tracker(self, idx, dataset):
        data = dataset[idx]
        cam_data = data['camera_data'].get("cam01")

        #extract the current frame and restart the tracker 
        frame_bgr = extract_frame(cam_data)
        bboxes = cam_data.get('2D_bboxes', [])
        entities = cam_data.get('entities', [])
        self.kcf_tracker.init_trackers(frame_bgr, bboxes, entities)
    
    def _run_loop(self, dataset):
        self.kcf_tracker = KCFTracker()
        tracker_initialized = False
        paused = False
        
        for idx in range(len(dataset)):
            if not paused:
                data = dataset[idx]
                
                #obtenemos la camara frontal
                cam_data = data['camera_data'].get("cam01")
                
                frame_bgr = extract_frame(cam_data)
                bboxes = cam_data.get('2D_bboxes', [])
                entities = cam_data.get('entities', [])
                
                if not tracker_initialized:

                    #inicializamos el tracker con las cajas delimitadoras actuales
                    self.kcf_tracker.init_trackers(frame_bgr, bboxes, entities)
                    output = self.kcf_tracker.draw(
                        frame_bgr, 
                        [(int(b[0]), int(b[1]), int(b[2]), int(b[3]), e) 
                         for b, e in zip(bboxes, entities)]
                    )
                    tracker_initialized = True
                else:
                    results = self.kcf_tracker.update(frame_bgr)

                    #dibujamos las cajas detectadas
                    output = self.kcf_tracker.draw(frame_bgr, results)
    
                    #dibujamos las cajas ground-truth
                    for bbox  in bboxes:
                        x1, y1, x2, y2 = map(int, bbox[:4])
                        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 1)
                
                cv2.imshow("KCF Tracking", output)
            
            key = cv2.waitKey(30 if not paused else 1) & 0xFF
            
            if key == 27 or key == ord('q'):
                cv2.destroyAllWindows()
                return
            elif key == ord('r'):
                if not paused:
                    self.restart_tracker(idx, dataset)
            elif key == ord(' '):
                paused = not paused
