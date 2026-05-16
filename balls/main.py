import cv2
import numpy as np
import json
from pathlib import Path
import random

save_path = Path(__file__).parent

cv2.namedWindow("Image", cv2.WINDOW_GUI_NORMAL)

clicked_positions = []
colors = []
sequence = []
guess = []
draw_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]

def on_click(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_positions.append((x, y))

def generate_sequence():
    global sequence, guess
    nums = [0, 1, 2, 3]
    random.shuffle(nums)
    sequence = [nums[:2], nums[2:]]
    print("Generated:")
    print(sequence)

def guess_sequence(current_sequence):
    if len(sequence) == 0:
        return
    print("Current:", current_sequence)

    if current_sequence == sequence:
        print("Correct")
    else:
        print("Wrong")
        print("Expected:", sequence)

cv2.setMouseCallback("Image", on_click)

capture = cv2.VideoCapture(0)

config_path = save_path / "config_balls.json"
if config_path.exists():
    with config_path.open("r") as f:
        js = json.load(f)
        colors = [(np.array(c[0], dtype="u1"), np.array(c[1], dtype="u1")) for c in js.get("colors", [])]

positions = [[], [], [], []]

while True:
    ret, frame = capture.read()
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("g"):
        generate_sequence()
    elif key == ord("r"):
        guess = []
        print("Guess reset")
    elif key == ord(" "):
        guess_sequence(current_sequence)

    while len(clicked_positions) > 0 and len(colors) < 4:
        x, y = clicked_positions.pop(0)
        color = hsv[y, x]

        lower = np.clip(color * 0.9, 0, 255).astype("u1")
        upper = np.clip(color * 1.1, 0, 255).astype("u1")
        upper[1] = 255
        upper[2] = 255

        colors.append((lower, upper))
        print(f"Color {len(colors)} added")

    combined_mask = np.zeros(frame.shape[:2], dtype="uint8")
    detected = []

    for i, (lower, upper) in enumerate(colors):
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype="u1"))

        combined_mask = cv2.bitwise_or(combined_mask, mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(contour)

            if radius > 10:
                current_ball = i
                center = (int(x), int(y))
                detected.append((center[0], center[1], i))
                draw_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
                cv2.circle(frame, center, int(radius), draw_colors[i], 3)
                cv2.circle(frame, center, 5, (255, 255, 255), -1)

                positions[i].append(center)
                if len(positions[i]) > 20:
                    positions[i].pop(0)

                for j, pos in enumerate(positions[i]):
                    cv2.circle(frame, pos, 2, draw_colors[i], -1)
    if len(detected) == 4:
        detected.sort(key=lambda p: p[1])
        top = detected[:2]
        bottom = detected[2:]
        top.sort(key=lambda p: p[0])
        bottom.sort(key=lambda p: p[0])
        ordered = top + bottom
        flat = [color_id for _, _, color_id in ordered]
        current_sequence = [flat[:2], flat[2:]]
    else:
        current_sequence = []

    cv2.imshow("Image", frame)
with config_path.open("w") as f:
    json.dump({
        "colors": [
            (lower.tolist(), upper.tolist()) for lower, upper in colors
        ]
    }, f)