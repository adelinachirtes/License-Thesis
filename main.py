from ultralytics import YOLO
from utils import get_device

if __name__ == "__main__":

    # Select device for training
    device = get_device()

    # If there is no pre-trained model, use YOLO's default
    PRE_TRAINED_MODEL = 'Yolo-Weights/yolov8n.pt'

    # Load a model
    #model = YOLO("yolov8s.yaml")  # pornește un model YOLOv8 nano fără greutăți preantrenate
    model = YOLO("runs/detect/train20/weights/best.pt")  # încarcă un model YOLOv8 cu greutăți preantrenate

    # Train the model
    results = model.train(data='config.yaml', epochs=300, imgsz=640, device=device, batch=8, workers=1, augment=True)

    model.val(data='config.yaml', split='test')