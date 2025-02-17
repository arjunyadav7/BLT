from ultralytics import YOLO
import cv2

# Load trained YOLOv8 model
model = YOLO("C:/Users/arjun/Desktop/Jupyter Notebook/Number plate detection/license_plate_model/weights/best.pt")  # Update path

# Load sample image
image_path = "photos/numPlate6.jpg"
image = cv2.imread(image_path)

# Perform inference
results = model(image)

# Display results
for result in results:
    # Boxes (xyxy), confidence scores, class labels
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        confidence = box.conf[0].item()  # Confidence score
        class_id = int(box.cls[0].item())  # Class ID

        # Draw bounding box and label
        label = f"{model.names[class_id]}: {confidence:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show output image
cv2.imshow("Detection Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
