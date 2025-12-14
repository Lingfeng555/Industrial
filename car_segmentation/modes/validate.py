##
## validate.py
## Driver code to validate the SVM classifier
##
## Diego Revilla (with the help of ChatGPT)
## Copyright (c) 2025 University of Deusto
##

from typing import List
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from src.once_dataset import ONCEDataset
from ..classifier import SVMClassifier
from ..data_utils import collect_samples


class ValidateMode:
    def __init__(self, data_path: str, model_path: str = None):
        self.data_path = data_path
        self.model_path = model_path
        self.classifier = SVMClassifier()
    
    def run(self):
        # Load trained model
        if not self.classifier.load(self.model_path):
            print("Error: Could not load model. Please train the model first using --train")
            return
        
        val_dataset = ONCEDataset(
            data_path=self.data_path,
            split='val',
            data_type="both",
            level="frame",
            logger_name="SVM_Validator",
            show_logs=True
        )
        
        images, y_true = collect_samples(val_dataset)
        
        y_pred = self.predict_batch(images)
        
        self._print_metrics(y_true, y_pred)
        self._plot_confusion_matrix(y_true, y_pred)

    def predict_batch(self, images: List[np.ndarray]):
        X = []
        for img in tqdm(images, desc="Extracting features"):
            feats = self.classifier.extract_features(img)
            X.append(feats)
        return list(self.classifier.clf.predict(X))
    
    def _print_metrics(self, y_true, y_pred):
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(classification_report(y_true, y_pred))
        
        print("\n" + "=" * 60)
        print("METRICS SUMMARY")
        print("=" * 60)
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(f"F1 Score (weighted): {f1_score(y_true, y_pred, average='weighted'):.4f}")
        print(f"F1 Score (macro): {f1_score(y_true, y_pred, average='macro'):.4f}")
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        print("\n" + "=" * 60)
        print("CONFUSION MATRIX")
        print("=" * 60)
        classes = sorted(list(set(y_true + list(y_pred))))
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        print(cm)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        print("\nConfusion matrix saved to confusion_matrix.png")
        plt.show()