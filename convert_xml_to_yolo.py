import os
import xml.etree.ElementTree as ET

# === CONFIG ===
xml_path = r"D:\Licenta\AI-Basketball-Shot-Detection-Tracker\yolo_dataset\annotations.xml"
output_path = r"D:\Licenta\AI-Basketball-Shot-Detection-Tracker\yolo_dataset\yolo_labels"
img_width = 1280
img_height = 720

os.makedirs(output_path, exist_ok=True)

tree = ET.parse(xml_path)
root = tree.getroot()

# Parsăm toate adnotările pentru mingea de baschet
tracks = root.findall(".//track[@label='ball']")
annotations = {}

for track in tracks:
    for box in track.findall("box"):
        frame = int(box.attrib["frame"])
        xtl = float(box.attrib["xtl"])
        ytl = float(box.attrib["ytl"])
        xbr = float(box.attrib["xbr"])
        ybr = float(box.attrib["ybr"])

        # Convertim în format YOLO (valori normalizate)
        x_center = ((xtl + xbr) / 2) / img_width
        y_center = ((ytl + ybr) / 2) / img_height
        width = (xbr - xtl) / img_width
        height = (ybr - ytl) / img_height

        line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

        # Adăugăm la fișierul corespunzător imaginii
        if frame not in annotations:
            annotations[frame] = []
        annotations[frame].append(line)

# Salvăm fiecare fișier .txt
for frame, lines in annotations.items():
    filename = f"frame_{frame:06d}.txt"
    filepath = os.path.join(output_path, filename)
    with open(filepath, "w") as f:
        f.write("\n".join(lines))

print("✅ Fișierele .txt în format YOLO au fost generate cu succes.")
