import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from typing import Literal
import os
import json
import time
from src.CustomLogger import CustomLogger
import numpy as np

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

        # Validate data directories
        camera_dirs = os.listdir(os.path.join(self.data_path, "camera"))
        lidar_dirs = os.listdir(os.path.join(self.data_path, "lidar"))
        anno_dirs = os.listdir(self.annotation_path)

        assert len(camera_dirs) == len(anno_dirs) == len(lidar_dirs), "Data and annotation directories count do not match."
        assert camera_dirs != None and anno_dirs != None and lidar_dirs != None, "Data or annotation directories are None."
        assert camera_dirs == anno_dirs == lidar_dirs, f"Data directories {camera_dirs} and annotation directories {anno_dirs} do not match."

        # Load camera data paths
        records = camera_dirs
        for record in records:
            record_path = os.path.join(self.data_path, "camera", record)
            sensors = os.listdir(record_path)
            for sensor in sensors:
                path = os.path.join(record_path, sensor)
                assert os.path.exists(path), f"Camera sensor path does not exist: {path}"
        self.cameras = sensors

        # Load lidar data paths
        records = lidar_dirs
        for record in records:
            path = os.path.join(self.data_path, "lidar", record, LIDAR_FOLDER_NAME)
            assert os.path.exists(path), f"Lidar path does not exist: {path}"
            self.lidar_paths.append(path)

        # Load the annotation paths
        for record in anno_dirs:
            path = os.path.join(self.annotation_path, record, f"{record}.json")
            assert os.path.exists(path), f"Annotation path does not exist: {path}"
            self.annotation_paths.append(path)

            with open(path, 'r') as f:
                annotation_dict = json.load(f)
                self.annotation_dicts[record] = annotation_dict

        # Precompute accumulative frame counts for indexing
        accumulative_count = 0
        for record in self.annotation_dicts.keys():
            accumulative_count += len(self.annotation_dicts[record]["frames"])
            self.accumulative_frame_counts.append( (record, accumulative_count) )

        # Precompute frame index mapping
        for index in range(self.accumulative_frame_counts[-1][1]):
            record, frame_info = self._find_record_and_frame(index)
            if 'annos' in frame_info.keys():  # filter out frames without annos information, not sure why they exist
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
        record, frame_info = self.frame_indexs[idx]
        if self.data_type in ["camera", "both"]:
            camera_data = {}
            for camera in self.cameras:
                frame_path = os.path.join(self.data_path, "camera", record, camera , frame_info["frame_id"] + ".jpg")
                data = {
                    "image_tensor": read_image(frame_path).float() / 255.0 if self.data_type in ["camera", "both"] else None,
                    "entities": frame_info["annos"]["names"],
                    "2D_bboxes": frame_info["annos"]["boxes_2d"][camera]
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
    
        