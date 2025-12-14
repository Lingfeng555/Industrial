##
## car_segmentation.py
## Entry point for the car segmentation project.
##
## Diego Revilla
## Copyright (c) 2025 University of Deusto
##

from car_segmentation.modes import TrainMode, ValidateMode, SegmentMode, TrackMode
from args import args

MODEL_PATH = "svm_classifier.pkl"

if __name__ == "__main__":
    if args.train:
        trainer = TrainMode(args.data_path, MODEL_PATH)
        trainer.run()
    elif args.val:
        validator = ValidateMode(args.data_path, MODEL_PATH)
        validator.run()
    elif args.segment:
        segmenter = SegmentMode(args.data_path, MODEL_PATH)
        segmenter.run()
    elif args.track:
        tracker = TrackMode(args.data_path)
        tracker.run()
