import os
import torch
import numpy as np
from torchvision.io import read_image
from typing import Tuple, Iterator
class Car:

    data_path: str
    camera_fold: str
    lidar_fold:str
    car_id: str
    cams: list[str]
    records: list[str]

    frame_time: int

    def __init__(self, car_id: str, data_path: str):
        self.data_path = os.path.join(os.path.expanduser(data_path), "val")
        self.camera_fold = os.path.join(self.data_path, "camera")
        self.lidar_fold = os.path.join(self.data_path, "lidar")
        self.car_id = car_id
        self.cams = sorted(os.listdir(os.path.join(self.camera_fold, car_id)))
        self.records = sorted(os.listdir(os.path.join(self.camera_fold, car_id, "cam01")))

        #remove the subfix
        self.records = [record.split(".")[0] for record in self.records]

        self.frame_time = 0

    def increase_frame_time(self):
        self.frame_time += 1
        if self.frame_time == len(self.records):
            self.frame_time = 0

    def get_info(self) -> Iterator[Tuple[dict, dict]]:
        record = self.records[self.frame_time]
        self.increase_frame_time()

        lidar_path = os.path.join(self.lidar_fold, self.car_id, "lidar_roof", record+".bin")
        lidar_data = {
            "points": torch.from_numpy(np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4))
        }

        images = {}
        for camera in self.cams:
            image_path = os.path.join(self.camera_fold, self.car_id, camera, record+".jpg")
            images[camera] = read_image(image_path).float() / 255.0

        return lidar_data, images

if __name__ == '__main__':
    Car(car_id="000027", data_path="~/Desktop/DATA")