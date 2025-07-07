import math
import numpy as np
import torch

def get_device():
    """Automatically select devices -> mps（Mac） -> cpu"""
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device


def score(ball_pos, hoop_pos):
    if len(ball_pos) < 2 or len(hoop_pos) == 0:
        return False

    # Rim position
    rim_y = hoop_pos[-1][0][1] - 0.5 * hoop_pos[-1][3]
    rim_x1 = hoop_pos[-1][0][0] - 0.4 * hoop_pos[-1][2]
    rim_x2 = hoop_pos[-1][0][0] + 0.4 * hoop_pos[-1][2]
    rebound_buffer = 10

    try:
        # Folosește o regresie de gradul 2 dacă ai puncte
        if len(ball_pos) >= 5:
            x_vals = [p[0][0] for p in ball_pos]
            y_vals = [p[0][1] for p in ball_pos]
            a, b, c = np.polyfit(x_vals, y_vals, 2)

            for x in np.linspace(rim_x1, rim_x2, 20):
                y = a * x**2 + b * x + c
                if rim_y - 15 <= y <= rim_y + 30:
                    return True

        # Altfel fallback: 2 puncte liniare
        x = []
        y = []
        for i in reversed(range(len(ball_pos))):
            if ball_pos[i][0][1] < rim_y:
                x.append(ball_pos[i][0][0])
                y.append(ball_pos[i][0][1])
                if i + 1 < len(ball_pos):
                    x.append(ball_pos[i + 1][0][0])
                    y.append(ball_pos[i + 1][0][1])
                break

        if len(x) > 1:
            m, b = np.polyfit(x, y, 1)
            predicted_x = (rim_y - b) / m
            if rim_x1 - rebound_buffer < predicted_x < rim_x2 + rebound_buffer:
                return True

    except:
        return False

    return False



# Detects if the ball is below the net - used to detect shot attempts
def is_ball_below_hoop(ball_trajectory, hoop_detections):
    y_limit = hoop_detections[-1][0][1] + 0.5 * hoop_detections[-1][3]
    return ball_trajectory[-1][0][1] > y_limit

def is_ball_above_hoop(ball_trajectory, hoop_detections):
    hx, hy = hoop_detections[-1][0]
    hw, hh = hoop_detections[-1][2], hoop_detections[-1][3]
    x, y = ball_trajectory[-1][0]
    in_x_range = hx - 4 * hw < x < hx + 4 * hw
    in_y_range = hy - 2 * hh < y < hy - 0.5 * hh
    return in_x_range and in_y_range


# Checks if center point is near the hoop
def is_near_hoop(center, hoop_pos):
    if len(hoop_pos) < 1:
        return False
    x = center[0]
    y = center[1]

    x1 = hoop_pos[-1][0][0] - 1 * hoop_pos[-1][2]
    x2 = hoop_pos[-1][0][0] + 1 * hoop_pos[-1][2]
    y1 = hoop_pos[-1][0][1] - 1 * hoop_pos[-1][3]
    y2 = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]

    if x1 < x < x2 and y1 < y < y2:
        return True
    return False


# Removes inaccurate data points
def refine_ball_positions(ball_pos, frame_count):
    # Removes inaccurate ball size to prevent jumping to wrong ball
    if len(ball_pos) > 1:
        # Width and Height
        w1 = ball_pos[-2][2]
        h1 = ball_pos[-2][3]
        w2 = ball_pos[-1][2]
        h2 = ball_pos[-1][3]

        # X and Y coordinates
        x1 = ball_pos[-2][0][0]
        y1 = ball_pos[-2][0][1]
        x2 = ball_pos[-1][0][0]
        y2 = ball_pos[-1][0][1]

        # Frame count
        f1 = ball_pos[-2][1]
        f2 = ball_pos[-1][1]
        f_dif = f2 - f1

        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        max_dist = 4 * math.sqrt((w1) ** 2 + (h1) ** 2)

        # Ball should not move a 4x its diameter within 5 frames
        if (dist > max_dist) and (f_dif < 3):
            ball_pos.pop()

        # Ball should be relatively square
        # elif (w2*1.4 < h2) or (h2*1.4 < w2):
        #     ball_pos.pop()

    # Remove points older than 30 frames
    if len(ball_pos) > 0:
        if frame_count - ball_pos[0][1] > 30:
            ball_pos.pop(0)

    return ball_pos


def refine_hoop_positions(hoop_pos):
    # Prevents jumping from one hoop to another
    if len(hoop_pos) > 1:
        x1 = hoop_pos[-2][0][0]
        y1 = hoop_pos[-2][0][1]
        x2 = hoop_pos[-1][0][0]
        y2 = hoop_pos[-1][0][1]

        w1 = hoop_pos[-2][2]
        h1 = hoop_pos[-2][3]
        w2 = hoop_pos[-1][2]
        h2 = hoop_pos[-1][3]

        f1 = hoop_pos[-2][1]
        f2 = hoop_pos[-1][1]

        f_dif = f2-f1

        dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)

        max_dist = 0.5 * math.sqrt(w1 ** 2 + h1 ** 2)

        # Hoop should not move 0.5x its diameter within 5 frames
        if dist > max_dist and f_dif < 5:
            hoop_pos.pop()

        # Hoop should be relatively square
        if (w2*1.3 < h2) or (h2*1.3 < w2):
            hoop_pos.pop()

    # Remove old points
    if len(hoop_pos) > 25:
        hoop_pos.pop(0)

    return hoop_pos
