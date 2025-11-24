"""
Multi-car tracking system using KCF tracker with semantic segmentation.
Optimized for 1-second frame intervals from front-facing camera.
"""
import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from args import args
from src.once_dataset import ONCEDataset


def create_kcf_tracker() -> Any:
    """Create KCF tracker with version compatibility."""
    try:
        # Try newer OpenCV API (4.5.1+)
        return cv2.legacy.TrackerKCF_create()
    except AttributeError:
        try:
            # Try older OpenCV API
            return cv2.TrackerKCF_create()
        except AttributeError:
            # Fallback to CSRT if KCF not available
            try:
                return cv2.legacy.TrackerCSRT_create()
            except AttributeError:
                return cv2.TrackerCSRT_create()


@dataclass
class TrackedCar:
    """Represents a tracked car with its state."""
    id: int
    tracker: Any  # KCF or CSRT tracker
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    lost_frames: int = 0
    color: Tuple[int, int, int] = None
    
    def __post_init__(self):
        if self.color is None:
            # Generate random color for visualization
            self.color = tuple(np.random.randint(0, 255, 3).tolist())


class CarTrackingSystem:
    """Multi-object tracking system for cars using KCF."""
    
    def __init__(self, 
                 max_lost_frames: int = 5,
                 min_confidence: float = 0.3,
                 iou_threshold: float = 0.3):
        """
        Initialize the tracking system.
        
        Args:
            max_lost_frames: Maximum frames a car can be lost before removal
            min_confidence: Minimum detection confidence to add new tracker
            iou_threshold: IoU threshold for matching detections to tracks
        """
        self.tracked_cars: List[TrackedCar] = []
        self.next_id = 0
        self.max_lost_frames = max_lost_frames
        self.min_confidence = min_confidence
        self.iou_threshold = iou_threshold
        
        # Simple car detector using background subtraction + morphology
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=50, detectShadows=True
        )
        
    def detect_cars(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect potential car regions using background subtraction.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            List of bounding boxes (x, y, w, h)
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame, learningRate=0.001)
        
        # Remove shadows (value 127)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        
        # More aggressive morphological operations to reduce noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size (car-like dimensions) - much stricter
        detections = []
        min_area = 3000  # Higher minimum for cars
        max_area = frame.shape[0] * frame.shape[1] * 0.15  # Max 15% of frame
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Stricter aspect ratio and size filters
                aspect_ratio = w / h if h > 0 else 0
                
                # Cars should be wider than tall typically
                if 0.8 < aspect_ratio < 2.5 and w > 60 and h > 40:
                    detections.append((x, y, w, h))
        
        return detections
    
    def calculate_iou(self, box1: Tuple[int, int, int, int], 
                      box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update_trackers(self, frame: np.ndarray) -> List[TrackedCar]:
        """
        Update all active trackers.
        
        Args:
            frame: Current BGR frame
            
        Returns:
            List of successfully tracked cars
        """
        cars_to_remove = []
        
        for car in self.tracked_cars:
            success, bbox = car.tracker.update(frame)
            
            if success:
                car.bbox = tuple(map(int, bbox))
                car.lost_frames = 0
                car.confidence = min(1.0, car.confidence + 0.1)
            else:
                car.lost_frames += 1
                car.confidence = max(0.0, car.confidence - 0.2)
                
                # Mark for removal if lost too long
                if car.lost_frames > self.max_lost_frames:
                    cars_to_remove.append(car)
        
        # Remove lost cars
        for car in cars_to_remove:
            self.tracked_cars.remove(car)
        
        return self.tracked_cars
    
    def add_new_trackers(self, frame: np.ndarray, detections: List[Tuple[int, int, int, int]]):
        """
        Add new trackers for unmatched detections.
        
        Args:
            frame: Current BGR frame
            detections: List of detected bounding boxes
        """
        for detection in detections:
            # Check if detection matches any existing tracker
            matched = False
            for car in self.tracked_cars:
                iou = self.calculate_iou(detection, car.bbox)
                if iou > self.iou_threshold:
                    matched = True
                    break
            
            # Add new tracker if no match found
            if not matched:
                tracker = create_kcf_tracker()
                tracker.init(frame, detection)
                
                new_car = TrackedCar(
                    id=self.next_id,
                    tracker=tracker,
                    bbox=detection,
                    confidence=0.8
                )
                self.tracked_cars.append(new_car)
                self.next_id += 1
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame: detect, track, and visualize.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Annotated frame with tracked cars
        """
        # Update existing trackers
        self.update_trackers(frame)
        
        # Detect new cars
        detections = self.detect_cars(frame)
        
        # Add new trackers for unmatched detections
        self.add_new_trackers(frame, detections)
        
        # Draw results
        output = self.draw_tracks(frame)
        
        return output
    
    def draw_tracks(self, frame: np.ndarray) -> np.ndarray:
        """Draw tracked cars on frame."""
        output = frame.copy()
        
        for car in self.tracked_cars:
            x, y, w, h = car.bbox
            
            # Draw bounding box
            thickness = 3 if car.confidence > 0.7 else 2
            cv2.rectangle(output, (x, y), (x + w, y + h), car.color, thickness)
            
            # Draw ID and confidence
            label = f"Car {car.id} ({car.confidence:.2f})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Background for text
            cv2.rectangle(output, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), car.color, -1)
            
            # Text
            cv2.putText(output, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw statistics
        stats = f"Tracking {len(self.tracked_cars)} cars"
        cv2.putText(output, stats, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return output
    
    def reset(self):
        """Reset all trackers and state."""
        self.tracked_cars.clear()
        self.next_id = 0
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=50, detectShadows=True
        )


def extract_frame(cam_data) -> np.ndarray:
    """Extract and convert frame from dataset to BGR format."""
    # Extract image tensor
    if isinstance(cam_data, dict):
        image_tensor = cam_data.get('image_tensor')
    else:
        image_tensor = cam_data
    
    # Convert tensor to numpy array
    if hasattr(image_tensor, 'numpy'):
        frame = image_tensor.numpy()
    else:
        frame = np.asarray(image_tensor)
    
    # Convert from CxHxW to HxWxC if needed
    if frame.ndim == 3 and frame.shape[0] in [1, 3, 4]:
        frame = frame.transpose(1, 2, 0)
    
    # Convert to uint8 if needed
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    return frame_bgr


def main():
    """Main execution function."""
    splits = os.listdir(args.data_path)
    
    # Initialize tracking system
    tracker_system = CarTrackingSystem(
        max_lost_frames=3,  # Lower due to 1-second gaps
        min_confidence=0.4,
        iou_threshold=0.3
    )
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing split: {split}")
        print(f"{'='*60}\n")
        
        # Load dataset
        dataset = ONCEDataset(
            data_path=args.data_path,
            split=split,
            data_type="both",
            level="frame",
            logger_name=f"ONCEDataset_{split}",
            show_logs=True
        )
        
        if len(dataset) == 0:
            print(f"Split '{split}' is empty. Skipping.")
            continue
        
        # Reset tracker for new split
        tracker_system.reset()
        
        print(f"\nProcessing {len(dataset)} frames...")
        print("Press 'q' or ESC to quit")
        print("Press 'r' to reset trackers")
        print("Press SPACE to pause/resume\n")
        
        paused = False
        
        for idx in range(len(dataset)):
            if not paused or cv2.waitKey(1) & 0xFF == ord(' '):
                data = dataset[idx]
                
                # Use front camera (cam01 or cam05)
                cam_data = data['camera_data'].get("cam01") or data['camera_data'].get("cam05")
                if cam_data is None:
                    continue
                
                # Extract frame
                frame_bgr = extract_frame(cam_data)
                
                # Process frame with tracking
                output = tracker_system.process_frame(frame_bgr)
                
                # Display
                cv2.imshow("Multi-Car Tracking (KCF)", output)
            
            # Handle keyboard input
            key = cv2.waitKey(30 if not paused else 1) & 0xFF
            
            if key == 27 or key == ord('q'):  # ESC or 'q' to quit
                print("\nExiting...")
                cv2.destroyAllWindows()
                return
            elif key == ord('r'):  # 'r' to reset
                print("Resetting trackers...")
                tracker_system.reset()
            elif key == ord(' '):  # SPACE to pause/resume
                paused = not paused
                print("Paused" if paused else "Resumed")
        
        cv2.destroyAllWindows()
        print(f"\nCompleted split: {split}")


if __name__ == "__main__":
    main()
