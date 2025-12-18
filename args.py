import argparse
import os

parser = argparse.ArgumentParser(description="Ejemplo de argumentos")

parser.add_argument("--data_path", type=str, help="Ruta del dataset ONCE (obligatorio)", default=os.path.expanduser("~/Desktop/DATA/ONCE"))

# Car segmentation modes
parser.add_argument('--train', action='store_true', help='Train the SVM classifier')
parser.add_argument('--val', action='store_true', help='Validate model and show metrics')
parser.add_argument('--segment', action='store_true', help='Run segmentation visualization')
parser.add_argument('--track', action='store_true', help='Run KCF tracking')

args, _ = parser.parse_known_args()