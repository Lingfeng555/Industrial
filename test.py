import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
sys.path.append('./yolov12')

from yolov12.ultralytics import YOLO
from args import args

from once_dataset import ONCEDataset

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

# Resize the image to 640x640 for YOLOv12 input
image_tensor = item["camera_data"]["cam01"]["image_tensor"] # this is a torch.Tensor
image_tensor = np.transpose(image_tensor.cpu().numpy(), (1, 2, 0))

# Get original dimensions
original_height, original_width = image_tensor.shape[:2]

# Resize image to 640x640
image_resized = cv2.resize(image_tensor, (640, 640))

# Calculate scaling factors
scale_x = 640 / original_width
scale_y = 640 / original_height

true_entities = item["camera_data"]["cam01"]["entities"]

# Resize the bboxes accordingly
bboxes = item["camera_data"]["cam01"]['2D_bboxes']
if bboxes is not None and len(bboxes) > 0:
    bboxes_resized = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox[:4]
        # Scale bbox coordinates
        x1_scaled = x1 * scale_x
        y1_scaled = y1 * scale_y
        x2_scaled = x2 * scale_x
        y2_scaled = y2 * scale_y
        # Keep other attributes if they exist
        if len(bbox) > 4:
            bboxes_resized.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled] + list(bbox[4:]))
        else:
            bboxes_resized.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled])
    bboxes = np.array(bboxes_resized)

# Resize the image to 640x640 for YOLOv12 input
results = model.predict(source=image_resized, imgsz=640, conf=0.25)
pred_entities = results[0].boxes.xyxy.cpu().numpy()
true_entities = item["camera_data"]["cam01"]["entities"]

# Plot the results
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Convert BGR to RGB for matplotlib
image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

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

