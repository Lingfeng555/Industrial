##
## __init__.py
## Exposure of modules outside the car_segmentation packages.
##
## Diego Revilla
## Copyright (c) 2025 University of Deusto
##

from .classifier import SVMClassifier
from .segmenter import CarSegmenter
from .tracker import KCFTracker
from .data_utils import extract_frame, map_label, collect_samples

__all__ = [
    'SVMClassifier',
    'CarSegmenter', 
    'KCFTracker',
    'extract_frame',
    'map_label',
    'collect_samples'
]
