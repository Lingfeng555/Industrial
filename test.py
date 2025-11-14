import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision.transforms import ToPILImage

sys.path.append('./yolov12')

from yolov12.ultralytics import YOLO
from args import args

from src.once_dataset import ONCEDataset

model = YOLO(model="yolov12m.pt")  
train_dataset = ONCEDataset(
            data_path=args.data_path,
            split="train",
            data_type="camera",
            level="frame",
            logger_name=f"ONCEDataset_train",
            show_logs=True
        )
item = train_dataset[random.randint(0, len(train_dataset)-1)]

# Parse to PIL
image_tensor = item["camera_data"]["cam01"]["image_tensor"] * 255 # this is a torch.Tensor
to_pil = ToPILImage()
pil_img = to_pil(image_tensor) 

true_entities = item["camera_data"]["cam01"]["entities"]

# Resize the bboxes accordingly
bboxes = item["camera_data"]["cam01"]['2D_bboxes']

# Resize the image to 640x640 for YOLOv12 input
results = model.predict(source=pil_img, imgsz=640, conf=0.25)
pred_entities = results[0].boxes.xyxy.cpu().numpy()
true_entities = item["camera_data"]["cam01"]["entities"]

# Plot the results
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Convert BGR to RGB for matplotlib
print("image_tensor shape:", image_tensor.shape)
image_rgb = image_tensor.permute(1, 2, 0).cpu().numpy() / 255.0

# Plot predicted bboxes
ax = axes[0]
ax.imshow(image_rgb)
ax.set_title("Predicted Bounding Boxes (YOLOv12m)")
ax.axis('off')

# Draw predicted bboxes
for bbox in pred_entities:
    x1, y1, x2, y2 = bbox[:4]
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)

# Plot true bboxes
ax = axes[1]
ax.imshow(image_rgb)
ax.set_title("Ground Truth Bounding Boxes")
ax.axis('off')

# Draw true bboxes if they exist
if bboxes is not None and len(bboxes) > 0:
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox[:4]
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='green', linewidth=2)
        ax.add_patch(rect)

plt.tight_layout()
plt.savefig('detection_results.png', dpi=150, bbox_inches='tight')
print("Results saved to 'detection_results.png'")
plt.show()

model.train(
    data='data.yaml',  
    epochs=10,  
    imgsz=640,  
    batch=2,  
    name='yolov12m_once_train'
)
