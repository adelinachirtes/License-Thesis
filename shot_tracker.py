# # Importăm librăriile necesare
# import cv2  # OpenCV pentru procesare video
# import cvzone  # Extensie pentru OpenCV (desene, colțuri, etc.)
# import math
# import numpy as np  # Pentru manipulare de matrici (imagini)
# from ultralytics import YOLO  # YOLO - model de detecție obiecte
#
# # Importăm funcții ajutătoare dintr-un fișier extern utils.py
# from utils import (
#     evaluate_shot,       # Verifică dacă o aruncare a fost reușită
#     passed_hoop_down,    # Verifică dacă mingea a trecut în jos prin coș
#     passed_hoop_up,      # Verifică dacă mingea a trecut în sus prin coș
#     near_hoop,           # Verifică dacă mingea este aproape de coș
#     smooth_hoop_path,    # Netezește traiectoria coșului
#     smooth_ball_path,    # Netezește traiectoria mingii
#     select_device        # Alege CPU sau GPU
# )
#
#
# class ShotTracker:
#     def __init__(self, video_path, model_path="best.pt"):
#         # Încarcă modelul YOLO
#         self.model = YOLO(model_path)
#         self.device = select_device()  # Selectează dispozitivul (CPU/GPU)
#         self.cap = cv2.VideoCapture(video_path)  # Deschide video-ul pentru procesare
#
#         # Etichetele claselor detectate de model
#         self.class_labels = ['Basketball', 'Basketball Hoop']
#
#         # Inițializări de variabile
#         self.frame_idx = 0  # Contor pentru cadre
#         self.current_frame = None
#
#         self.ball_trajectory = []  # Salvează traiectoria mingii
#         self.hoop_detections = []  # Salvează detectările coșului
#
#         self.total_makes = 0  # Coșuri reușite
#         self.total_attempts = 0  # Încercări totale
#
#         # Zone logice pentru detecția unei aruncări
#         self.in_up_zone = False
#         self.in_down_zone = False
#         self.up_frame_idx = 0
#         self.down_frame_idx = 0
#
#         # Variabile pentru mesajele vizuale de feedback
#         self.feedback_duration = 20  # Durata mesajului
#         self.feedback_timer = 0
#         self.overlay_msg = "Waiting..."
#         self.feedback_color = (0, 0, 0)  # Negru
#
#         # Pornește procesarea video
#         self._process_video()
#
#
#     def _process_video(self):
#         # Rulează frame cu frame video-ul
#         while True:
#             ret, self.current_frame = self.cap.read()  # Citește un cadru
#             if not ret:
#                 break  # Sfârșitul videoclipului
#
#             # Trimite cadrul la modelul YOLO pentru detecție
#             predictions = self.model(self.current_frame, stream=True, device=self.device)
#
#             # Extrage mingea și coșul din rezultat
#             self._extract_detections(predictions)
#
#             # Desenează traiectorii netezite
#             self._refine_and_draw()
#
#             # Verifică dacă a fost o aruncare și dacă a fost reușită
#             self._evaluate_shot_attempts()
#
#             # Afișează scorul și mesajele vizuale
#             self._render_feedback()
#
#             self.frame_idx += 1  # Trecem la următorul cadru
#             cv2.imshow('Shot Tracker', self.current_frame)  # Afișează fereastra video
#
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break  # Ieșire din program apăsând tasta 'q'
#
#         self.cap.release()
#         cv2.destroyAllWindows()  # Curăță ferestrele la final
#
#
#     def _extract_detections(self, results):
#         # Parcurge fiecare rezultat din YOLO
#         for result in results:
#             for box in result.boxes:
#                 # Coordonatele colțurilor
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 width, height = x2 - x1, y2 - y1
#                 conf = round(float(box.conf[0]), 2)  # Încrederea detecției
#                 label = int(box.cls[0])  # Clasa (minge sau coș)
#                 center = (x1 + width // 2, y1 + height // 2)  # Centru obiect
#
#                 # Dacă este minge și are încredere suficientă
#                 if self.class_labels[label] == "Basketball" and (
#                     conf > 0.3 or (near_hoop(center, self.hoop_detections) and conf > 0.15)
#                 ):
#                     # Salvează în traiectorie
#                     self.ball_trajectory.append((center, self.frame_idx, width, height, conf))
#                     # Desenează un dreptunghi colțuros pe imagine
#                     cvzone.cornerRect(self.current_frame, (x1, y1, width, height))
#
#                 # Dacă este coș
#                 elif self.class_labels[label] == "Basketball Hoop" and conf > 0.5:
#                     self.hoop_detections.append((center, self.frame_idx, width, height, conf))
#                     cvzone.cornerRect(self.current_frame, (x1, y1, width, height))
#
#
#     def _refine_and_draw(self):
#         # Netezește traiectoria mingii
#         self.ball_trajectory = smooth_ball_path(self.ball_trajectory, self.frame_idx)
#
#         # Desenează fiecare punct din traiectoria mingii
#         for point in self.ball_trajectory:
#             cv2.circle(self.current_frame, point[0], 2, (0, 0, 255), 2)  # Roșu
#
#         # Dacă există suficiente detectări ale coșului, netezește
#         if len(self.hoop_detections) > 1:
#             self.hoop_detections = smooth_hoop_path(self.hoop_detections)
#             # Desenează punctul actual al coșului
#             cv2.circle(self.current_frame, self.hoop_detections[-1][0], 2, (128, 128, 0), 2)
#
#
#     def _evaluate_shot_attempts(self):
#         # Asigură-te că ai traiectorie și coș
#         if self.ball_trajectory and self.hoop_detections:
#             # Verifică dacă mingea a trecut în sus prin coș
#             if not self.in_up_zone:
#                 self.in_up_zone = passed_hoop_up(self.ball_trajectory, self.hoop_detections)
#                 if self.in_up_zone:
#                     self.up_frame_idx = self.ball_trajectory[-1][1]
#
#             # Verifică dacă apoi a trecut în jos
#             if self.in_up_zone and not self.in_down_zone:
#                 self.in_down_zone = passed_hoop_down(self.ball_trajectory, self.hoop_detections)
#                 if self.in_down_zone:
#                     self.down_frame_idx = self.ball_trajectory[-1][1]
#
#             # Dacă a trecut în sus și apoi în jos => a fost o aruncare
#             if self.frame_idx % 10 == 0 and self.in_up_zone and self.in_down_zone and self.up_frame_idx < self.down_frame_idx:
#                 self.total_attempts += 1  # Incrementăm încercările
#                 self.in_up_zone = self.in_down_zone = False  # Resetăm zonele
#
#                 # Verificăm dacă a fost un coș valid
#                 if evaluate_shot(self.ball_trajectory, self.hoop_detections):
#                     self.total_makes += 1
#                     self.feedback_color = (0, 255, 0)  # Verde
#                     self.overlay_msg = "Scored"
#                 else:
#                     self.feedback_color = (0, 0, 255)  # Roșu
#                     self.overlay_msg = "Missed"
#
#                 self.feedback_timer = self.feedback_duration  # Pornim timer pentru mesaj
#
#
#     def _render_feedback(self):
#         # Afișează scorul total pe imagine
#         display_text = f"{self.total_makes} / {self.total_attempts}"
#         cv2.putText(self.current_frame, display_text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
#         cv2.putText(self.current_frame, display_text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)
#
#         # Afișează mesaj de tip "Scored" sau "Missed" dacă este activ
#         if self.feedback_timer > 0:
#             (text_width, _), _ = cv2.getTextSize(self.overlay_msg, cv2.FONT_HERSHEY_SIMPLEX, 3, 6)
#             text_x = self.current_frame.shape[1] - text_width - 40
#             text_y = 100
#             cv2.putText(self.current_frame, self.overlay_msg, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3, self.feedback_color, 6)
#
#             # Creează un efect vizual de fade cu culoarea mesajului
#             alpha = 0.2 * (self.feedback_timer / self.feedback_duration)
#             faded = np.full_like(self.current_frame, self.feedback_color)
#             self.current_frame = cv2.addWeighted(self.current_frame, 1 - alpha, faded, alpha, 0)
#             self.feedback_timer -= 1
#
#
# # Pornim programul dacă rulăm fișierul
# if __name__ == "__main__":
#     ShotTracker()
