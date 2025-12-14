import cv2
import numpy as np
from typing import List, Optional
from .classifier import SVMClassifier


class CarSegmenter:    
    def __init__(
        self, 
        k: int = 2, 
        low_threshold: int = 50, 
        high_threshold: int = 150
    ):
        """
        Initialize the car segmenter.
        
        Args:
            k: Number of clusters for K-means
            low_threshold: Lower threshold for Canny edge detection
            high_threshold: Upper threshold for Canny edge detection
        """
        self.k = k
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    def process_roi(self, roi: np.ndarray) -> np.ndarray:
        """
        Process a region of interest using edge detection and K-means clustering.
        
        Args:
            roi: BGR image array of the region of interest
            
        Returns:
            Binary segmentation mask as BGR image
        """
        if roi.size == 0:
            return roi
            
        # 1. Canny Edge Detection (Line separation)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold)
        
        # Erosion on lines
        kernel = np.ones((3, 3), np.uint8)
        eroded_edges = cv2.erode(edges, kernel, iterations=1)
        
        # 2. K-Means Clustering
        pixel_values = roi.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            pixel_values, self.k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        # 3. Get most common color and binarize
        labels = labels.flatten()
        counts = np.bincount(labels)
        most_common_label = np.argmax(counts)
        
        # Create binary mask: 255 for most common label, 0 otherwise
        mask = np.zeros_like(labels, dtype=np.uint8)
        mask[labels == most_common_label] = 255
        mask = mask.reshape(roi.shape[:2])
        
        # Remove pixels corresponding to eroded edges
        mask[eroded_edges == 255] = 0
        
        # Convert to BGR for visualization
        binary_roi = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        return binary_roi
    
    def visualize(
        self, 
        frame: np.ndarray, 
        bboxes: List, 
        entities: Optional[List] = None, 
        classifier: Optional[SVMClassifier] = None
    ) -> np.ndarray:
        """
        Visualize segmentation results on a frame.
        
        Args:
            frame: BGR image array
            bboxes: List of bounding boxes [x1, y1, x2, y2, ...]
            entities: List of entity labels (optional)
            classifier: SVMClassifier instance for prediction overlay (optional)
            
        Returns:
            Annotated frame with segmented ROIs
        """
        output = frame.copy()
        h_frame, w_frame = frame.shape[:2]
        frame_area = h_frame * w_frame
        
        if entities is None:
            entities = [""] * len(bboxes)
        
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
            
            roi = frame[y1:y2, x1:x2]
            segmented_roi = self.process_roi(roi)
            
            output[y1:y2, x1:x2] = segmented_roi
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            text = label
            if classifier:
                pred = classifier.predict(roi)
                text = f"GT:{label} | Pred:{pred}"
            
            cv2.putText(
                output, text, (x1, max(y1 - 5, 10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
            
        return output
