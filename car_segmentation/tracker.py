import cv2
import numpy as np
from typing import List, Tuple


class KCFTracker:
    """Multi-object tracker using KCF (Kernelized Correlation Filters)."""
    
    def __init__(self):
        """Initialize the KCF tracker."""
        self.trackers: List[cv2.Tracker] = []
        self.labels: List[str] = []
    
    def init_trackers(
        self, 
        frame: np.ndarray, 
        bboxes: List, 
        entities: List[str]
    ) -> None:
        """
        Initialize trackers for all detected objects in a frame.
        
        Args:
            frame: BGR image array
            bboxes: List of bounding boxes [x1, y1, x2, y2, ...]
            entities: List of entity labels
        """
        self.trackers = []
        self.labels = []
        h_frame, w_frame = frame.shape[:2]
        frame_area = h_frame * w_frame
        
        for bbox, label in zip(bboxes, entities):
            if bbox[0] < 0:
                continue
                
            x1, y1, x2, y2 = map(int, bbox[:4])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_frame, x2)
            y2 = min(h_frame, y2)
            
            if x1 >= x2 or y1 >= y2:
                continue
            
            # Ignore if bbox takes more than 30% of screen space
            box_area = (x2 - x1) * (y2 - y1)
            if box_area > 0.3 * frame_area:
                continue
            
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
            self.trackers.append(tracker)
            self.labels.append(label)
    
    def update(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, str]]:
        """
        Update all trackers with a new frame.
        
        Args:
            frame: BGR image array
            
        Returns:
            List of tuples (x1, y1, x2, y2, label) for successfully tracked objects
        """
        results = []
        for tracker, label in zip(self.trackers, self.labels):
            success, box = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                results.append((x, y, x + w, y + h, label))
        return results
    
    def draw(
        self, 
        frame: np.ndarray, 
        results: List[Tuple[int, int, int, int, str]]
    ) -> np.ndarray:
        """
        Draw tracking results on a frame.
        
        Args:
            frame: BGR image array
            results: List of tuples (x1, y1, x2, y2, label)
            
        Returns:
            Annotated frame with tracking boxes
        """
        output = frame.copy()
        for (x1, y1, x2, y2, label) in results:
            cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                output, f"KCF: {label}", (x1, max(y1 - 5, 10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
            )
        return output
    
    @property
    def num_trackers(self) -> int:
        """Return the number of active trackers."""
        return len(self.trackers)
