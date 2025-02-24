# step 1: Import the ultralytics

from ultralytics import YOLO

# Step 2: Define paths of your dataset

dataset_path = "Dataset" 

# Step 3: Initialize the YOLOv8 model
model = YOLO("yolov8n.pt")  

# Step 4: Train the model
results = model.train(
    data=f"{dataset_path}/data.yaml",  # Path of your dataset YAML file
    epochs=50,                         # Number of epochs for training
    imgsz=640,                         # Image size for training
    batch=16,                          # No.of batches
)

# Save the model after training
model.save("yolov8_trained_model.pt")


