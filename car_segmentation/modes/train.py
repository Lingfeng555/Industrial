##
## train.py
## Driver code to train the SVM classifier
##
## Diego Revilla
## Copyright (c) 2025 University of Deusto
##

from src.once_dataset import ONCEDataset
from ..classifier import SVMClassifier
from ..data_utils import collect_samples

class TrainMode:
    def __init__(self, data_path: str, model_path: str = None, sample_ratio: float = 0.5):
        self.data_path = data_path
        self.model_path = model_path
        self.sample_ratio = sample_ratio
        self.classifier = SVMClassifier()
    
    def run(self):        
        train_dataset = ONCEDataset(
            data_path=self.data_path,
            split='train',
            data_type="both",
            level="frame",
            logger_name="SVM_Trainer",
            show_logs=True
        )
        
        images, labels = collect_samples(train_dataset, sample_ratio=self.sample_ratio)
        self.classifier.train(images, labels)
        self.classifier.save(self.model_path)
