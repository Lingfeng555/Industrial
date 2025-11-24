import os
import cv2
import numpy as np
from typing import Optional, Tuple
from args import args
from src.once_dataset import ONCEDataset

class CamShiftTracker:
    def __init__(self, hist_bins: Tuple[int, int] = (180, 256)):
        self.hist_bins = hist_bins
        self.roi_hist: Optional[np.ndarray] = None
        self.track_window: Optional[Tuple[int, int, int, int]] = None
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.rot_rect = None  # rotated rectangle from CamShift

    def initialize(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> Tuple[int, int, int, int]:
        # Expect BGR frame (OpenCV format)
        if bbox is None:
            bbox = tuple(map(int, cv2.selectROI("Select ROI", frame, False, False)))
            cv2.destroyWindow("Select ROI")
        x, y, w, h = bbox
        self.track_window = (x, y, w, h)

        roi = frame[y : y + h, x : x + w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # mask to remove low-saturation/value pixels
        mask = cv2.inRange(hsv_roi, np.array((0, 60, 32)), np.array((180, 255, 255)))

        # 2D histogram for H and S
        roi_hist = cv2.calcHist([hsv_roi], [0, 1], mask, list(self.hist_bins), [0, 180, 0, 256])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        self.roi_hist = roi_hist.astype(np.float32)
        self.rot_rect = None
        return self.track_window

    def update(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if self.roi_hist is None or self.track_window is None:
            return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        backproj = cv2.calcBackProject([hsv], [0, 1], self.roi_hist, [0, 180, 0, 256], scale=1)
        # apply CamShift to get the new location and rotated rectangle
        ret, new_window = cv2.KLC(backproj, self.track_window, self.term_crit)
        self.track_window = new_window
        self.rot_rect = ret
        return new_window

    def draw_box(self, frame: np.ndarray, color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
        if self.track_window is None:
            return frame
        out = frame.copy()
        if self.rot_rect is not None:
            pts = cv2.boxPoints(self.rot_rect)
            pts = np.int32(pts)  # Must be int32, not int8
            cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
        else:
            x, y, w, h = self.track_window
            cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        return out

if __name__ == "__main__":
    splits = os.listdir(args.data_path)

    for split in splits:
        dataset = ONCEDataset(
            data_path=args.data_path,
            split=split,
            data_type="both",
            level="frame",
            logger_name=f"ONCEDataset_{split}",
            show_logs=True
        )

        tracker = CamShiftTracker()

        for idx in range(20, len(dataset)):
            data = dataset[idx]
            cam_data = data['camera_data']["cam03"]

            # Extract image tensor and convert to numpy
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

            if idx == 20:
                tracker.initialize(frame_bgr)
            else:
                tracker.update(frame_bgr)

            output_frame = tracker.draw_box(frame_bgr)
            cv2.imshow("CamShift Tracking", output_frame)

            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC to exit
                break
            if key in (ord('i'), ord('I')):  # reinitialize ROI on 'I' or 'i'
                tracker.initialize(frame_bgr)

        cv2.destroyAllWindows()
