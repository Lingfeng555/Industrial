from torch.utils.data import Dataset
from typing import Literal
import os
import json

from src.CustomLogger import CustomLogger

ANNOTATION_FOLDER_NAME = "annotation"
LIDAR_FOLDER_NAME = "lidar_roof"

class ONCEDataset(Dataset):

    data_path: str
    annotation_path: str
    level : Literal["frame", "record"]
    data_type : Literal["camera", "lidar"]
    split: Literal["train", "val", "test"]
    
    logger: CustomLogger

    cameras_paths: list[str]
    lidar_paths: list[str]
    annotation_paths: list[str]
    
    annotation_dicts: list[dict]

    def __init__(self, 
                 data_path: str, 
                 split: Literal["train", "val", "test"], 
                 data_type: Literal["camera", "lidar", "both"],
                 level: Literal["frame", "record"], 
                 logger_name: str = "ONCEDataset", 
                 show_logs: bool = True):

        # Initialize dataset paths and logger
        self.data_path = os.path.join(data_path, split)
        self.annotation_path = os.path.join(data_path, split, ANNOTATION_FOLDER_NAME)
        self.split = split
        self.data_type = data_type
        self.level = level
        self.logger = CustomLogger(logger_name, show_logs=show_logs).get_logger()
        self.cameras_paths = []
        self.lidar_paths = []
        self.annotation_paths = []
        self.annotation_dicts = []

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
                self.cameras_paths.append(path)

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
                self.annotation_dicts.append(annotation_dict)

        for path in self.cameras_paths:
            self.logger.info(msg = f"Found camera data path: {path}")
        for path in self.lidar_paths:
            self.logger.info(msg = f"Found lidar data path: {path}")
        for path in self.annotation_paths:
            self.logger.info(msg = f"Found annotation path: {path}")

        self.logger.info(msg = f"ONCEDataset(data_path={self.data_path}, annotation_path={self.annotation_path})")

    def __len__(self):
        # Retorna el tamaño del dataset
        return 1000  # Valor de ejemplo

    def __getitem__(self, idx):
        # Retorna un ítem del dataset
        return {"data": f"Sample {idx} from {self.split}"}  # Valor de ejemplo