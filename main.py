import os
import polars as pl
import argparse
    
from args import args
from once_dataset import ONCEDataset

def main(args: argparse.Namespace):
    splits = os.listdir(args.data_path)
    print("Cargando datos desde:", splits)

    dataset = ONCEDataset(data_path=args.data_path, split=splits[0], data_type="camera", level="record")
    dataset = ONCEDataset(data_path=args.data_path, split=splits[0], data_type="lidar", level="frame")
    ...

if __name__ == "__main__":
    main(args)