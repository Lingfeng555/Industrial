##
## classifier.py
## Exposure of modules outside the modes package.
##
## Diego Revilla
## Copyright (c) 2025 University of Deusto
##

import cv2
import pickle
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


class SVMClassifier:    
    DEFAULT_MODEL_PATH = "svm_classifier.pkl"
    
    def __init__(self, C = 10):
        self.clf = make_pipeline(StandardScaler(), LinearSVC(random_state=42, C=C))
        self.hog = cv2.HOGDescriptor(
            (64, 64), # tamaño de ventana
            (16, 16), # tamaño del bloque
            (8, 8),   # stride del bloque
            (8, 8),   # tamaño de celda
            16        # número de bins
        )
        self.classes = []

    def extract_features(self, img):
        #convierte la imagen a escala de grises y computamos el histograma de gradientes orientados (HOG)
        resized = cv2.resize(img, (64, 64))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        features = self.hog.compute(gray)
        return features.flatten()

    def train(self, images, labels):
        #entrenamos el clasificador SVM con las características extraídas de las imágenes
        X = []
        y = []

        for img, label in tqdm(zip(images, labels), total=len(images), desc="Extracting features"):
            feats = self.extract_features(img)
            X.append(feats)
            y.append(label)

        self.classes = list(set(y))
        self.clf.fit(X, y)

    def predict(self, img):
        #predecimos la clase de una imagen extrayendo sus características y usando el clasificador entrenado
        feats = self.extract_features(img)
        return self.clf.predict([feats])[0]
    
    def save(self, path = None):
        #guardamos los pesos en un archivo
        path = path or self.DEFAULT_MODEL_PATH
        with open(path, 'wb') as f:
            pickle.dump({
                'clf': self.clf,
                'classes': self.classes
            }, f)
    
    def load(self, path = None):
        #cargamos los pesos entrenados
        path = path or self.DEFAULT_MODEL_PATH
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.clf = data['clf']
                self.classes = data.get('classes', [])
            return True
        except FileNotFoundError:
            print(f"Model file not found: {path}")
            return False
