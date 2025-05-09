# import torch
#
# print("Torch version:", torch.__version__)
# print("Torch file location:", torch.__file__)
# print("CUDA available:", torch.cuda.is_available())
# print("CUDA version:", torch.version.cuda)
# print("cuDNN version:", torch.backends.cudnn.version())
# print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
import torch
print(torch.__version__)                # Should show 2.6.0+cu118
print(torch.__file__)                   # Should be inside your project venv path
print(torch.cuda.is_available())        # Should be True
print(torch.cuda.get_device_name(0))    # Should print your GPU
