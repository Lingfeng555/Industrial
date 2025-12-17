# Industrial - YOLOv12 Object Detection

Detección de objetos en tiempo real usando YOLOv12 con el dataset ONCE.

## Datasets

- **ONCE Dataset**: [https://once-for-auto-driving.github.io/](https://once-for-auto-driving.github.io/)
- **Paper**: [ONCE: One Million Scenarios for Autonomous Driving](https://arxiv.org/abs/2106.11037)
- **YOLOv12**: [GitHub - sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12)

## Instalación

1. **Instalar FlashAttention** (opcional, para mejor rendimiento en GPU)
```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

2. **Instalar dependencias**
```bash
pip install -r requirements.txt
pip install -e .
```

3. **Clonar YOLOv12** (si no está incluido)
```bash
git clone https://github.com/sunsmarterjie/yolov12.git
```

## Uso

### Para finetunear Yolo
```bash
python test.py
```

### Antes de arrancar streamlit crear imagenes con el modelo
```bash
python inference.py
```

### Para arrancar la aplicacion de streamlit

```bash
bentoml serve service.py:YoloService --port 3000
bentoml serve service.py:ImageServerService --port 3001
streamlit run app.py
```


