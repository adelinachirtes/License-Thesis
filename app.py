# import os
# from flask import Flask, render_template, redirect, url_for
# from shot_tracker import Shot  # asigură-te că fișierul se numește așa și e în același director
#
# app = Flask(__name__)
#
# # Calea unde sunt video-urile
# VIDEO_DIR = "D:/Licenta/AI-Basketball-Shot-Detection-Tracker"
#
# @app.route("/")
# def index():
#     videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]
#     return render_template("index.html", videos=videos)
#
# @app.route("/detect/<video_name>")
# def detect(video_name):
#     video_path = os.path.join(VIDEO_DIR, video_name)
#     if os.path.exists(video_path):
#         print(f"Rulez detectia pe: {video_path}")
#         ShotTracker(video_path=video_path)  # rulează detecția aici
#         return f"<h2>Detecția a fost rulată pentru {video_name}. Închide fereastra video pentru a reveni.</h2>"
#     else:
#         return f"<h2>Fișierul {video_name} nu a fost găsit.</h2>", 404
#
# if __name__ == "__main__":
#     app.run(debug=True)
from ultralytics import YOLO
import torch

print(torch.cuda.is_available())  # ar trebui să fie True

model = YOLO("yolov8n.pt")
results = model.train(data='Dataset/data.yaml', device=0)  # 0 înseamnă primul GPU
