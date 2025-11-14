import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from typing import Literal
import os
import json
import time
from CustomLogger import CustomLogger
import numpy as np
from torchvision.transforms import ToPILImage
import shutil

ANNOTATION_FOLDER_NAME = "annotation"
LIDAR_FOLDER_NAME = "lidar_roof"

class ONCEDataset(Dataset):

    # Builder variables
    data_path: str
    annotation_path: str
    level : Literal["frame", "record"]
    data_type : Literal["camera", "lidar"]
    split: Literal["train", "val", "test"]
    
    # Internal variables
    cameras: list[str]
    lidar_paths: list[str]
    annotation_paths: list[str]
    annotation_dicts: dict[str, dict]
    accumulative_frame_counts: list[tuple[str, int]]
    frame_indexs: list[tuple[str, dict]]

    # Internal variables
    logger: CustomLogger

    # Meta data
    image_dimension: tuple = (1920, 1020)
    time_labels: list[str] = ["morning", "noon", "afternoon", "night"]
    weather_labels: list[str] = ["sunny", "cloudy", "rainy"]
    area_labels: list[str] = ["downtown", "suburbs","tunnel","highway","bridge"]
    entity_labels: list[str] = ['Car', 'Truck', 'Bus', 'Pedestrian', 'Cyclist']

    def __init__(self, 
                 data_path: str, 
                 split: Literal["train", "val", "test"], 
                 data_type: Literal["camera", "lidar", "both"],
                 level: Literal["frame", "record"], 
                 logger_name: str = "ONCEDataset", 
                 show_logs: bool = True):
        time_start = time.time()

        # Initialize dataset paths and logger
        self.data_path = os.path.join(data_path, split)
        self.annotation_path = os.path.join(data_path, split, ANNOTATION_FOLDER_NAME)
        self.split = split
        self.data_type = data_type
        self.level = level
        self.logger = CustomLogger(logger_name, show_logs=show_logs).get_logger()
        self.lidar_paths = []
        self.annotation_paths = []
        self.annotation_dicts = {}
        self.accumulative_frame_counts = []
        self.frame_indexs = []

        # Validate and sort data directories to ensure consistent ordering
        camera_dirs = sorted(os.listdir(os.path.join(self.data_path, "camera")))
        lidar_dirs = sorted(os.listdir(os.path.join(self.data_path, "lidar")))
        anno_dirs = sorted(os.listdir(self.annotation_path))

        if not (len(camera_dirs) and len(lidar_dirs) and len(anno_dirs)):
            raise ValueError(f"Missing data directories under {self.data_path}. camera:{camera_dirs}, lidar:{lidar_dirs}, anno:{anno_dirs}")

        if not (len(camera_dirs) == len(anno_dirs) == len(lidar_dirs)):
            raise AssertionError("Data and annotation directories count do not match.")

        # Load camera data paths: determine sensors from the first record and validate existence across records
        records = camera_dirs
        first_record = records[0]
        first_record_path = os.path.join(self.data_path, "camera", first_record)
        sensors = sorted(os.listdir(first_record_path))
        if not sensors:
            raise ValueError(f"No camera sensors found in {first_record_path}")

        for record in records:
            for sensor in sensors:
                path = os.path.join(self.data_path, "camera", record, sensor)
                assert os.path.exists(path), f"Camera sensor path does not exist: {path}"
        self.cameras = sensors

        # Load lidar data paths (sorted)
        for record in lidar_dirs:
            path = os.path.join(self.data_path, "lidar", record, LIDAR_FOLDER_NAME)
            assert os.path.exists(path), f"Lidar path does not exist: {path}"
            self.lidar_paths.append(path)

        # Load the annotation paths (sorted)
        for record in anno_dirs:
            path = os.path.join(self.annotation_path, record, f"{record}.json")
            assert os.path.exists(path), f"Annotation path does not exist: {path}"
            self.annotation_paths.append(path)

            with open(path, 'r') as f:
                annotation_dict = json.load(f)
                self.annotation_dicts[record] = annotation_dict

        # Precompute accumulative frame counts for indexing (use sorted order of records)
        accumulative_count = 0
        for record in sorted(self.annotation_dicts.keys()):
            accumulative_count += len(self.annotation_dicts[record]["frames"])
            self.accumulative_frame_counts.append( (record, accumulative_count) )

        # Precompute frame index mapping
        if not self.accumulative_frame_counts:
            # No frames found in annotations
            self.logger.warning(msg=f"No frames found in annotations for {self.data_path}")
        else:
            total_frames = self.accumulative_frame_counts[-1][1]
            for index in range(total_frames):
                record, frame_info = self._find_record_and_frame(index)
                # filter out frames without annos information
                if isinstance(frame_info, dict) and 'annos' in frame_info and frame_info['annos'] is not None:
                    self.frame_indexs.append( (record, frame_info) )

        time_end = time.time()
        self.logger.info(msg = f"ONCEDataset(data_path={self.data_path}, annotation_path={self.annotation_path}, level={self.level}, len={self.__len__()}); initialized in {time_end - time_start:.2f} seconds.")

    def __len__(self):
        return len(self.frame_indexs)
                
    def _find_record_and_frame(self, idx: int) -> tuple[str, dict]:
        previous_accum_count = 0
        for record, accum_count in self.accumulative_frame_counts:
            if idx < accum_count:
                relative_idx = idx - previous_accum_count
                #print(record, relative_idx)
                return record, self.annotation_dicts[record]["frames"][relative_idx]
            previous_accum_count = accum_count
        raise IndexError(f"Index {idx} out of range for dataset")

    def frame_by_index(self, idx: int) -> tuple[str, dict]:
        ret = {}
        if not self.frame_indexs:
            raise IndexError(f"Dataset contains no frame-level entries (len=0); cannot access index {idx}.")
        if idx < 0 or idx >= len(self.frame_indexs):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.frame_indexs)} frames.")

        record, frame_info = self.frame_indexs[idx]
        if self.data_type in ["camera", "both"]:
            camera_data = {}
            for camera in self.cameras:
                frame_path = os.path.join(self.data_path, "camera", record, camera , frame_info["frame_id"] + ".jpg")
                data = {
                    "image_tensor": read_image(frame_path).float() / 255.0 if self.data_type in ["camera", "both"] else None,
                    "entities": frame_info["annos"]["names"],
                    "2D_bboxes": frame_info["annos"]["boxes_2d"][camera],
                    "position": frame_info["pose"]
                }
                camera_data[camera] = data
            ret["camera_data"] = camera_data
        if self.data_type in ["lidar", "both"]:
            lidar_path = os.path.join(self.data_path, "lidar", record, LIDAR_FOLDER_NAME , frame_info["frame_id"] + ".bin")
            lidar_data = {
                "points": torch.from_numpy(np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)) if self.data_type in ["lidar", "both"] else None,
                "3D_bboxes": frame_info["annos"]["boxes_3d"]
            }
            ret["lidar_data"] = lidar_data

        ret["metadata"] = self.annotation_dicts[record]["meta_info"]
        ret["calibration"] = self.annotation_dicts[record]["calib"]
        ret["entities"] = frame_info["annos"]["names"]
        return ret
    
    def __getitem__(self, idx):
        if self.level == "frame":
            return self.frame_by_index(idx)
        elif self.level == "record": # Return frame data along with next frame index for a future feasible dreamer
            return (self.frame_by_index(idx), self.frame_indexs[idx+1])
        else:
            raise ValueError(f"Invalid level: {self.level}")


from multiprocessing import Pool, cpu_count

# Globals used inside worker processes
_dataset = None
_categories = None
_image_dim = None
_to_pil = None

def _init_worker(dataset, categories):
    """Initializer for each worker process."""
    global _dataset, _categories, _image_dim, _to_pil
    _dataset = dataset
    _categories = categories
    _image_dim = dataset.image_dimension  # (W, H)
    _to_pil = ToPILImage()

def _export_single_frame(args):
    """
    Worker function.
    Processes a single (dataset_idx, cam) pair and writes one image + one label file.
    """
    frame_idx, ds_idx, cam, images_dir, labels_dir = args

    item = _dataset[ds_idx]
    cam_data = item["camera_data"][cam]

    image_tensor = cam_data["image_tensor"]      # torch.Tensor [C,H,W]
    entities = cam_data["entities"]              # list of class names
    bboxes = cam_data["2D_bboxes"]               # list/array of [x1,y1,x2,y2,...]

    # Save image
    img_path = os.path.join(images_dir, f"{frame_idx:06d}.jpg")
    pil_img = _to_pil(image_tensor)
    pil_img.save(img_path)

    # Save label
    label_path = os.path.join(labels_dir, f"{frame_idx:06d}.txt")
    W, H = _image_dim  # assuming (width, height)

    lines = []
    for entity, bbox in zip(entities, bboxes):
        if bbox[0] == -1:
            continue
        x1, y1, x2, y2 = bbox[:4]
        class_id = _categories.index(entity)

        x_center = (x1 + x2) / 2 / W
        y_center = (y1 + y2) / 2 / H
        width = (x2 - x1) / W
        height = (y2 - y1) / H

        lines.append(f"{class_id} {x_center} {y_center} {width} {height}\n")

    # Write once per file
    if lines:
        with open(label_path, "w") as f:
            f.writelines(lines)

def export_dataset_to_yolo_mp(
    dataset,
    indices,
    images_dir,
    labels_dir,
    categories,
    num_workers=None,
):
    """
    Export given indices of a dataset to YOLO format using multiprocessing.
    One frame = one (dataset_idx, cam) pair -> one image + one txt.
    """
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    # Precompute tasks with unique frame_idx
    tasks = []
    frame_idx = 0
    for ds_idx in indices:
        for cam in dataset.cameras:
            tasks.append((frame_idx, ds_idx, cam, images_dir, labels_dir))
            frame_idx += 1

    # Run in parallel
    with Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(dataset, categories),
    ) as pool:
        for _ in pool.imap_unordered(_export_single_frame, tasks):
            pass  # you could add a tqdm, logging, etc. here

class Once_yolo_dataset:

    data_path: str
    logger: CustomLogger
    categories: list[str]
    
    def __init__(self, data_path, logger_name="Once_yolo_dataset", show_logs=True, num_workers=os.cpu_count()-1):
        self.data_path = data_path
        self.logger = CustomLogger(logger_name, show_logs=show_logs).get_logger()
        self.categories = ['Car', 'Truck', 'Bus', 'Pedestrian', 'Cyclist']
        self.num_workers = num_workers

        if "yolo" not in os.listdir(self.data_path):
            self.logger.warning(msg=f"No 'yolo' folder found in {self.data_path}. Creating one.")

            yolo_split_path = os.path.join(self.data_path, "yolo")
            os.makedirs(yolo_split_path, exist_ok=True)

            # Build original datasets
            train_dataset = ONCEDataset(
                data_path=self.data_path,
                split="train",
                data_type="camera",
                level="frame",
                logger_name=f"ONCEDataset_train",
                show_logs=show_logs
            )
            eval_dataset = ONCEDataset( 
                data_path=self.data_path,
                split="val",
                data_type="camera",
                level="frame",
                logger_name=f"ONCEDataset_eval",
                show_logs=show_logs
            )

            images_folder_path = os.path.join(yolo_split_path, "images")
            labels_folder_path = os.path.join(yolo_split_path, "labels")
            os.makedirs(images_folder_path, exist_ok=True)
            os.makedirs(labels_folder_path, exist_ok=True)

            # ----------------- VAL SPLIT (eval_dataset) -----------------
            val_images_path = os.path.join(images_folder_path, "val")
            val_labels_path = os.path.join(labels_folder_path, "val")

            eval_indices = list(range(len(eval_dataset)))
            self.logger.info(msg=f"Exporting val split to YOLO format at {val_images_path} and {val_labels_path}...")
            export_dataset_to_yolo_mp(
                dataset=eval_dataset,
                indices=eval_indices,
                images_dir=val_images_path,
                labels_dir=val_labels_path,
                categories=self.categories,
                num_workers=self.num_workers
            )

            # ----------------- TRAIN / TEST SPLIT (from train_dataset) -----------------
            all_train_idx = list(range(len(train_dataset)))
            np.random.shuffle(all_train_idx)
            split_idx = int(0.8 * len(all_train_idx))
            train_indices = all_train_idx[:split_idx]
            test_indices = all_train_idx[split_idx:]

            # TRAIN
            train_images_path = os.path.join(images_folder_path, "train")
            train_labels_path = os.path.join(labels_folder_path, "train")
            self.logger.info(msg=f"Exporting train split to YOLO format at {train_images_path} and {train_labels_path}...")
            export_dataset_to_yolo_mp(
                dataset=train_dataset,
                indices=train_indices,
                images_dir=train_images_path,
                labels_dir=train_labels_path,
                categories=self.categories,
                num_workers=self.num_workers
            )

            # TEST
            test_images_path = os.path.join(images_folder_path, "test")
            test_labels_path = os.path.join(labels_folder_path, "test")
            self.logger.info(msg=f"Exporting test split to YOLO format at {test_images_path} and {test_labels_path}...")
            export_dataset_to_yolo_mp(
                dataset=train_dataset,
                indices=test_indices,
                images_dir=test_images_path,
                labels_dir=test_labels_path,
                categories=self.categories,
                num_workers=self.num_workers
            )

if __name__ == "__main__":
    dataset = Once_yolo_dataset(
        data_path="/home/lingfeng/Desktop/DATA/ONCE/",
        logger_name="Once_yolo_dataset",
        show_logs=True
    )
    #shutil.rmtree("/home/lingfeng/Desktop/DATA/ONCE/yolo")