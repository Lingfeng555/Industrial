import pickle
import joblib
import bentoml
import os

MODEL_PATH = "processing/model.pkl" 
MODEL_NAME = "speed_regression_model" # Nombre que tendrá en BentoML
TOP_7_FEATURES = ['speed_y_avg_20', 'speed_x', 'speed_x_avg_20', 'lidar_n_points_total', 'lidar_intensity_mean', 'lidar_n_points_under_100m', 'lidar_points_10_20m'] # las 7 variables para entrenar el modelo

print(f"Importando el modelo '{MODEL_PATH}' a BentoML...")

# Cargar el modelo .pkl desde el disco
if not os.path.exists(MODEL_PATH):
    print(f"Error: No se encuentra el archivo del modelo en '{MODEL_PATH}'")
    print("Por favor, descarga el .pkl de tu Colab y colócalo en esta carpeta.")
else:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    # Guardar el modelo en el Model Store de BentoML
    bento_model = bentoml.sklearn.save_model(
        MODEL_NAME,
        model,
        metadata={
            "model_type": "RandomForestRegressor",
            "feature_names": TOP_7_FEATURES
        }
    )
    
    print(f"Modelo guardado en BentoML: {bento_model}")

