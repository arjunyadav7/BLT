from ultralytics import YOLO

# Loading the model
model = YOLO("yolov8n.pt")  # Using a pretrained model instead of yaml

absolute_path = "C:/Users/arjun/Desktop/Jupyter Notebook/Number plate detection"
# Using the model
results = model.train(data="num_plate_config.yaml", epochs=10, project=absolute_path, name="license_plate_model")
