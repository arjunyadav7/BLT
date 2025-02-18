# pre-trained YOLOv8m-seg model for segmenting cars & buses
from ultralytics import YOLO
import cv2
import numpy as np

# Load the pre-trained YOLOv8 segmentation model trained on COCO dataset
model = YOLO("yolov8n-seg.pt")

image_path = "Models/Car-Bus segmentation/sample images/traffic1.jpg"
image = cv2.imread(image_path)
height, width = image.shape[:2]
new_width = width // 2
new_height = height // 2
image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

results = model(image)

# COCO class IDs: car=2, bus=5
selected_classes = {2: (0, 255, 0), 5: (255, 0, 0)}

# Create an overlay to draw masks without modifying the original image
overlay = np.zeros_like(image, dtype=np.uint8)

for result in results:
    for mask, class_id, box in zip(result.masks.data, result.boxes.cls, result.boxes.xyxy):
        class_id = int(class_id)
        if class_id in selected_classes:
            color = selected_classes[class_id]  # Get corresponding color
            
            # Convert segmentation mask from tensor to numpy array
            mask = mask.cpu().numpy()  # Ensure mask is on CPU
            mask = (mask * 255).astype(np.uint8)  # Convert to 0-255 scale
            
            # Resize mask to match the original image dimensions
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Create a boolean mask where pixels are 255 (object) or 0 (background)
            mask_binary = mask > 128  # Threshold mask
            
            # Apply the mask color only where mask_binary is True
            overlay[mask_binary] = color  # Fill the segmentation area with color

            # Compute centroid for label placement
            x1, y1, x2, y2 = map(int, box)  # Get bounding box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center of bounding box
            label = "Car" if class_id == 2 else "Bus"
            cv2.putText(overlay, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Blend the overlay with the original image
final_image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

# Show the image with segmentation results
cv2.imshow("Segmented Cars and Buses", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
