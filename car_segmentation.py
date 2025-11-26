import os
import cv2
import numpy as np
from args import args
from src.once_dataset import ONCEDataset
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class SVMClassifier:
    def __init__(self):
        self.clf = make_pipeline(StandardScaler(), LinearSVC(random_state=42, tol=1e-5))
        self.hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
        self.is_trained = False

    def extract_features(self, img):
        if img.size == 0: return None
        img = cv2.resize(img, (64, 64))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = self.hog.compute(gray)
        return features.flatten()

    def train(self, images, labels):
        print("Extracting features...")
        X = []
        y = []
        for img, label in zip(images, labels):
            feats = self.extract_features(img)
            if feats is not None:
                X.append(feats)
                y.append(label)
        
        if len(set(y)) < 2:
            print(f"Not enough classes to train SVM. Found: {set(y)}")
            return
            
        print(f"Training SVM on {len(X)} samples with classes: {set(y)}")
        self.clf.fit(X, y)
        self.is_trained = True
        print("SVM training complete.")

    def predict(self, img):
        if not self.is_trained: return "Untrained"
        feats = self.extract_features(img)
        if feats is None: return "Error"
        return self.clf.predict([feats])[0]


class CarSegmenter:
  
    def __init__(self, k: int = 2, low_threshold: int = 50, high_threshold: int = 150):
        self.k = k
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    def process_roi(self, roi: np.ndarray) -> np.ndarray:
        if roi.size == 0:
            return roi
            
        # 1. Canny Edge Detection (Line separation)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
        edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold)
        
        # Erosion on lines
        kernel = np.ones((3,3), np.uint8)
        eroded_edges = cv2.erode(edges, kernel, iterations=1)
        
        # 2. K-Means Clustering
        pixel_values = roi.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixel_values, self.k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
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
    
    def visualize(self, frame: np.ndarray, bboxes: list, entities: list = None, classifier=None) -> np.ndarray:
        output = frame.copy()
        h_frame, w_frame = frame.shape[:2]
        frame_area = h_frame * w_frame
        
        if entities is None:
            entities = [""] * len(bboxes)
        
        for bbox, label in zip(bboxes, entities):
            if bbox[0] < 0: continue
            
            x1, y1, x2, y2 = map(int, bbox[:4])
            
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w_frame, x2); y2 = min(h_frame, y2)
            
            if x1 >= x2 or y1 >= y2: continue
            
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
            if classifier and classifier.is_trained:
                pred = classifier.predict(roi)
                text = f"GT:{label} | Pred:{pred}"
            
            cv2.putText(output, text, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        return output


def extract_frame(cam_data) -> np.ndarray:
    image_tensor = cam_data.get('image_tensor')
    frame = np.asarray(image_tensor)
    frame = frame.transpose(1, 2, 0)
    frame = (frame * 255).astype(np.uint8)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame_bgr


def main():
    splits = os.listdir(args.data_path)
    
    # Train SVM first
    classifier = SVMClassifier()
    if splits:
        print("Collecting training data for SVM...")
        # Use first split for training
        train_dataset = ONCEDataset(
            data_path=args.data_path,
            split=splits[1],
            data_type="both",
            level="frame",
            logger_name="SVM_Trainer",
            show_logs=False
        )
        images = []
        labels = []
        count = 0
        
        for i in tqdm(range(len(train_dataset))):
            data = train_dataset[i]
            cam_data = data['camera_data'].get("cam01") or data['camera_data'].get("cam05")
            if not cam_data: continue
            
            frame = extract_frame(cam_data)
            bboxes = cam_data.get('2D_bboxes', [])
            entities = cam_data.get('entities', [])
            
            for bbox, label in zip(bboxes, entities):
                if bbox[0] < 0: continue
                x1, y1, x2, y2 = map(int, bbox[:4])
                h, w = frame.shape[:2]
                x1=max(0,x1); y1=max(0,y1); x2=min(w,x2); y2=min(h,y2)
                if x1>=x2 or y1>=y2: continue
                
                roi = frame[y1:y2, x1:x2]
                
                # Map labels
                if label in ['Pedestrian', 'Cyclist']:
                    label = 'Other'
                
                images.append(roi)
                labels.append(label)
                count += 1
        
        if images:
            classifier.train(images, labels)
    
    segmenter = CarSegmenter(k=2, low_threshold=50, high_threshold=150)
    
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
        print("Press '+' to increase K")
        print("Press '-' to decrease K")
        print("Press 'w'/'s' for Canny Low")
        print("Press 'e'/'d' for Canny High")
        print("Press SPACE to pause/resume\n")
        
        paused = False
        
        for idx in range(len(dataset)):
            if not paused:
                data = dataset[idx]
                
                cam_data = data['camera_data'].get("cam01") or data['camera_data'].get("cam05")
                if cam_data is None:
                    continue
                
                frame_bgr = extract_frame(cam_data)
                bboxes = cam_data.get('2D_bboxes', [])
                entities = cam_data.get('entities', [])
                
                output = segmenter.visualize(frame_bgr, bboxes, entities, classifier)
                
                cv2.imshow("Car Segmentation", output)
            
            key = cv2.waitKey(30 if not paused else 1) & 0xFF
            
            if key == 27 or key == ord('q'):
                print("\nExiting...")
                cv2.destroyAllWindows()
                return
            elif key == ord('+') or key == ord('='):
                segmenter.k = min(10, segmenter.k + 1)
                print(f"K: {segmenter.k}")
            elif key == ord('-') or key == ord('_'):
                segmenter.k = max(2, segmenter.k - 1)
                print(f"K: {segmenter.k}")
            elif key == ord('w'):
                segmenter.low_threshold = min(segmenter.high_threshold, segmenter.low_threshold + 10)
                print(f"Canny: {segmenter.low_threshold}/{segmenter.high_threshold}")
            elif key == ord('s'):
                segmenter.low_threshold = max(0, segmenter.low_threshold - 10)
                print(f"Canny: {segmenter.low_threshold}/{segmenter.high_threshold}")
            elif key == ord('e'):
                segmenter.high_threshold = min(255, segmenter.high_threshold + 10)
                print(f"Canny: {segmenter.low_threshold}/{segmenter.high_threshold}")
            elif key == ord('d'):
                segmenter.high_threshold = max(segmenter.low_threshold, segmenter.high_threshold - 10)
                print(f"Canny: {segmenter.low_threshold}/{segmenter.high_threshold}")
            elif key == ord(' '):
                paused = not paused
                print("Paused" if paused else "Resumed")
        
        cv2.destroyAllWindows()
        print(f"\nCompleted split: {split}")


if __name__ == "__main__":
    main()
