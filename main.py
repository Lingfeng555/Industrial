import os
import time
from args import args
from src.once_dataset import ONCEDataset
from src.visualizer import ONCEVisualizer


def main():
    #visualizer = ONCEVisualizer()
    
    splits = os.listdir(args.data_path)
    
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
        
        time_start = time.time()
        #visualizer.visualize_split(dataset, split)
        time_end = time.time()
        
        print(f"\nProcessed {len(dataset)} frames in {time_end - time_start:.2f} seconds.")


if __name__ == "__main__":
    main()