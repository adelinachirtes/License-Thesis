import sys
print(sys.executable)
import torch
print("Torch version:", torch.__version__)
print("Torch CUDA available:", torch.cuda.is_available())
print("Torch CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())


# import torch
# from ultralytics import YOLO
#
# # Verifică dacă GPU este disponibil
# if torch.cuda.is_available():
#     device = 'cuda'
#     gpu_name = torch.cuda.get_device_name(0)
# else:
#     device = 'cpu'
#     gpu_name = 'No GPU available'
#
# print(f"Running on: {device.upper()}")
# print(f"GPU: {gpu_name}")
#
# # Încarcă modelul pe dispozitivul disponibil
# model = YOLO('yolov8n.pt')
# model.to(device)
#
# # Rulează o predicție de test (doar ca să forțeze folosirea dispozitivului)
# results = model.predict(source='https://ultralytics.com/images/bus.jpg', device=device, save=False)
#
# print("Predicția s-a realizat cu succes.")
