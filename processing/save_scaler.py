import pickle
import joblib
import bentoml
import os

SCALER_PATH = "processing/scaler.pkl"
SCALER_NAME = "speed_scaler" 

print(f"Guardando el scaler como '{SCALER_NAME}'...")

if os.path.exists(SCALER_PATH):
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    # Guardar escaler
    bento_scaler = bentoml.picklable_model.save_model(SCALER_NAME, scaler)
    
    print(f"Scaler (picklable) guardado en BentoML: {bento_scaler}")
else:
    print(f"Error: No se encuentra '{SCALER_PATH}'")