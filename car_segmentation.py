import os
import cv2
import numpy as np
from args import args
from src.once_dataset import ONCEDataset


class EdgeDetector:
  
    def __init__(self, low_threshold: int = 50, high_threshold: int = 150):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    def detect_edges(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold)
        
        return edges
    
    def visualize(self, frame: np.ndarray) -> tuple:
        edges = self.detect_edges(frame)
        
        edge_overlay = np.zeros_like(frame)
        edge_overlay[edges == 255] = [0, 0, 255]
        output = cv2.addWeighted(frame, 0.7, edge_overlay, 0.3, 0)
        
        return output, edges
    
    def set_thresholds(self, low: int, high: int):
        self.low_threshold = low
        self.high_threshold = high


def extract_frame(cam_data) -> np.ndarray:
    image_tensor = cam_data.get('image_tensor')
    frame = np.asarray(image_tensor)
    frame = frame.transpose(1, 2, 0)
    frame = (frame * 255).astype(np.uint8)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame_bgr


def main():
    splits = os.listdir(args.data_path)
    
    detector = EdgeDetector(low_threshold=50, high_threshold=150)
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing split: {split}")
        print(f"{'='*60}\n")
        
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
        
        print(f"\nProcessing {len(dataset)} frames...")
        print("Press 'q' or ESC to quit")
        print("Press '+' to increase sensitivity (lower threshold)")
        print("Press '-' to decrease sensitivity (higher threshold)")
        print("Press SPACE to pause/resume\n")
        
        paused = False
        
        for idx in range(len(dataset)):
            if not paused:
                data = dataset[idx]
                
                cam_data = data['camera_data'].get("cam01") or data['camera_data'].get("cam05")
                if cam_data is None:
                    continue
                
                frame_bgr = extract_frame(cam_data)
                

                output, _ = detector.visualize(frame_bgr)
                
                cv2.imshow("Canny Edge Detection (Overlay)", output)
            
            key = cv2.waitKey(30 if not paused else 1) & 0xFF
            
            if key == 27 or key == ord('q'):
                print("\nExiting...")
                cv2.destroyAllWindows()
                return
            elif key == ord('+') or key == ord('='):
                detector.low_threshold = max(10, detector.low_threshold - 10)
                detector.high_threshold = max(30, detector.high_threshold - 10)
                print(f"Thresholsds: {detector.low_threshold}/{detector.high_threshold} (more sensitive)")
            elif key == ord('-') or key == ord('_'):
                detector.low_threshold = min(150, detector.low_threshold + 10)
                detector.high_threshold = min(250, detector.high_threshold + 10)
                print(f"Thresholds: {detector.low_threshold}/{detector.high_threshold} (less sensitive)")
            elif key == ord(' '):
                paused = not paused
                print("Paused" if paused else "Resumed")
        
        cv2.destroyAllWindows()
        print(f"\nCompleted split: {split}")


if __name__ == "__main__":
    main()
