import os
import shutil
from sklearn.model_selection import train_test_split
from glob import glob

# === CONFIG ===
base_path = r"D:\Licenta\AI-Basketball-Shot-Detection-Tracker\yolo_dataset"
boxes_path = os.path.join(base_path, "boxes")
labels_path = os.path.join(base_path, "yolo_labels")  # asigură-te că .txt-urile sunt aici

# Unde se vor copia imaginile și etichetele
img_train = os.path.join(base_path, "images", "train")
img_val = os.path.join(base_path, "images", "val")
label_train = os.path.join(base_path, "labels", "train")
label_val = os.path.join(base_path, "labels", "val")

# Creează directoarele dacă nu există
for folder in [img_train, img_val, label_train, label_val]:
    os.makedirs(folder, exist_ok=True)

# Găsește toate imaginile PNG din boxes/
all_images = sorted(glob(os.path.join(boxes_path, "*.PNG")))

# Split 80% train / 20% val
train_imgs, val_imgs = train_test_split(all_images, test_size=0.2, random_state=42)

def copy_with_labels(image_list, dest_img_folder, dest_label_folder):
    for img_path in image_list:
        filename = os.path.basename(img_path)
        base = os.path.splitext(filename)[0]
        label_file = f"{base}.txt"

        # Copiază imaginea
        shutil.copy(img_path, os.path.join(dest_img_folder, filename))

        # Copiază și eticheta corespunzătoare (dacă există)
        label_src = os.path.join(labels_path, label_file)
        if os.path.exists(label_src):
            shutil.copy(label_src, os.path.join(dest_label_folder, label_file))

# Execută copierea
copy_with_labels(train_imgs, img_train, label_train)
copy_with_labels(val_imgs, img_val, label_val)

print("✅ Dataset structurat cu succes pentru YOLOv8!")
