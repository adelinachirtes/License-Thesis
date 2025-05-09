import os
import glob
import json

LABELS_PATH = "Dataset/train/labels"
OUTPUT_FILE = "../data/trajectories.json"
MIN_SEQ_LEN = 15
MAX_SEQ_LEN = 50


def get_label_files(path):
    return sorted(glob.glob(os.path.join(path, "*.txt")))


def extract_ball_trajectory():
    os.makedirs("data", exist_ok=True)
    all_files = get_label_files(LABELS_PATH)
    data = []
    current_seq = []

    for i, file in enumerate(all_files):
        with open(file, "r") as f:
            lines = f.readlines()

        ball_coords = None
        for line in lines:
            cls_id, x, y = line.strip().split()[:3]
            if int(cls_id) == 0:  # presupunem că mingea e clasa 0
                ball_coords = [float(x), float(y)]
                break

        if ball_coords:
            current_seq.append(ball_coords)

        if len(current_seq) >= MAX_SEQ_LEN or i == len(all_files) - 1:
            if len(current_seq) >= MIN_SEQ_LEN:
                data.append({
                    "trajectory": current_seq[:MAX_SEQ_LEN],
                    "label": None  # aici adaugi manual 0 sau 1
                })
            current_seq = []

    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[✓] Salvat {len(data)} traiectorii în {OUTPUT_FILE}")


if __name__ == "__main__":
    extract_ball_trajectory()
