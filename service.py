from __future__ import annotations

import bentoml
import numpy as np
import pickle


# Cargar runner del modelo 
SCALER_TAG = "speed_scaler:latest"

@bentoml.service()
class SpeedPredictionService:
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        print("Cargando el scaler desde el Model Store...")
        self.scaler = bentoml.picklable_model.load_model(SCALER_TAG)
        regressor_path = "processing/model.pkl"
        with open(regressor_path, "rb") as f:
            self.model = pickle.load(f)

        print("¡Scaler cargado!")

    @bentoml.api
    def predict_speed(self, input_data: np.ndarray) -> np.ndarray:
        features = self.scaler.transform(input_data)
        return self.model.predict(features)