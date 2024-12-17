from ultralytics import YOLO
model_object = YOLO("yolo11n.pt")
model_object.train(data = "./custom_data.yaml")
