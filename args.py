import argparse
import os

parser = argparse.ArgumentParser(description="Ejemplo de argumentos")

parser.add_argument("--data_path", type=str, help="Ruta del dataset ONCE (obligatorio)", default=os.path.expanduser("data"))

args = parser.parse_args()