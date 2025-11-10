import os
import argparse
import time
from args import args
from once_dataset import ONCEDataset

def main(args: argparse.Namespace):
    splits = os.listdir(args.data_path)
    print("Cargando datos desde:", splits)

    for split in splits:
        dataset = ONCEDataset(
            data_path=args.data_path,
            split=split,
            data_type="both",
            level="frame",
            logger_name=f"ONCEDataset_{split}",
            show_logs=True
        )
        time_start = time.time()
        for i in range(len(dataset)):
            data = dataset[i]
        time_end = time.time()
        print(f"Procesado {len(dataset)} frames del split '{split}' en {time_end - time_start:.2f} segundos.")
    ...

if __name__ == "__main__":
    main(args)