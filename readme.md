# Industrial - YOLOv12 Object Detection

Detección de objetos en tiempo real usando YOLOv12 con el dataset ONCE.

2. **Instalar dependencias**
```bash
pip install -r requirements.txt
pip install -e .
```

### Para arrancar la aplicacion de streamlit

```bash
bentoml serve service.py:SpeedPredictionService --port 3002 &
streamlit run app.py -- --data_path [PATH-DE-LOS-DATOS]
```

### Para usar solo el dashboard de procesamiento

```bash
cd processing
bentoml serve service.py:svc --port 3000
streamlit run app_final.py
```


