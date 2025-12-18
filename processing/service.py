import bentoml
import numpy as np
from bentoml.io import NumpyNdarray
from bentoml import Runnable


# Cargar runner del modelo 
model_ref = bentoml.sklearn.get("speed_regression_model:latest")
model_runner = model_ref.to_runner()
# Runner personalizado
SCALER_TAG = "speed_scaler:latest"
class ScalerRunnable(Runnable):
    # obligatorio para BentoML runner/strategy
    SUPPORTED_RESOURCES = ("cpu",)
    # indicar capacidades de concurrencia que BentoML consulta
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        print("Cargando el scaler desde el Model Store...")
        self.scaler = bentoml.picklable_model.load_model(SCALER_TAG)
        print("¡Scaler cargado!")

    # @bentoml.runnable.method define una función que el runner puede llamar
    @Runnable.method(batchable=True, batch_dim=0)
    def transform(self, input_data: np.ndarray) -> np.ndarray:
        return self.scaler.transform(input_data)


# Crear el scaler_runner a partir de nuestra clase personalizada
scaler_runner = bentoml.Runner(ScalerRunnable)
svc = bentoml.Service("predictor", runners=[model_runner, scaler_runner])


# Endpoint de la API
sample_array_input =[[-0.013326, 0.012518, 0.012518, 62883, 0.587167, 61538, 9267]]

@svc.api(input=NumpyNdarray.from_sample(sample_array_input), output=NumpyNdarray())

async def predict_speed(input_data: np.ndarray) -> np.ndarray:
    # primero Escalar los datos
    scaled_data = await scaler_runner.transform.async_run(input_data)
    # Predecir con los datos escalados
    prediction = await model_runner.predict.async_run(scaled_data)
    
    return prediction