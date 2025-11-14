# Industrial - YOLOv12 Object Detection

Detección de objetos en tiempo real usando YOLOv12 con el dataset ONCE.

## Datasets

- **ONCE Dataset**: [https://once-for-auto-driving.github.io/](https://once-for-auto-driving.github.io/)
- **Paper**: [ONCE: One Million Scenarios for Autonomous Driving](https://arxiv.org/abs/2106.11037)
- **YOLOv12**: [GitHub - sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12)

## Instalación

### Requisitos previos
- Python 3.11
- CUDA 11.x (opcional pero recomendado)
- conda o pip

### Pasos de instalación

1. **Crear entorno conda**
```bash
conda create -n yolov12 python=3.11
conda activate yolov12
```

2. **Instalar FlashAttention** (opcional, para mejor rendimiento en GPU)
```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
pip install -e .
```

4. **Clonar YOLOv12** (si no está incluido)
```bash
git clone https://github.com/sunsmarterjie/yolov12.git
```

## Uso

### Ejecutar inferencia en test
```bash
python test.py
```

Esto cargará una imagen aleatoria del dataset ONCE y mostrará:
- Predicciones de YOLOv12m (bboxes rojos)
- Ground truth annotations (bboxes verdes)

### Ejecutar con Gradio UI
```bash
python yolov12/app.py
```

## Estructura del proyecto

```
Industrial/
├── yolov12/              # Módulos de YOLOv12
│   ├── ultralytics/      # Código del modelo
│   └── app.py            # Interfaz Gradio
├── src/
│   └── once_dataset.py   # Dataset ONCE
├── test.py               # Script de prueba
├── args.py               # Configuración
├── requirements.txt      # Dependencias
└── readme.md             # Este archivo
```

## Configuración

Edita `args.py` para configurar:
- `data_path`: Ruta al dataset ONCE
- Otros parámetros de entrenamiento/inferencia

## Notas

- FlashAttention es opcional pero acelera significativamente la inferencia en GPUs modernas (Ampere+)
- Sin FlashAttention, el modelo usa `scaled_dot_product_attention` estándar de PyTorch
- Compatible con GPU y CPU (CPU es mucho más lento)

## Troubleshooting

Si tienes problemas con la importación de módulos:
```bash
# Asegúrate de que estás en el directorio correcto
cd /path/to/Industrial
python test.py
```

