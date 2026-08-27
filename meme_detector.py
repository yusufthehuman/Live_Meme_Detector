import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import time
from collections import deque

# -------------------------------------------------------------
# 1. Configuration & Setup
# -------------------------------------------------------------
MODEL_PATH = "emotion_model_pytorch_5.pth" #write the model name that u trained previously
PHONE_IP = "Your own phone ip adress"  #This is the usual IP in the droid app
TARGET_SIZE = (224, 224) #Image size
PANEL_SIZE = (480, 480)  # (width, height) for split screen panels
WINDOW_NAME = "Emotion Meme Detector"
FREEZE_DURATION = 2.0  # Duration in seconds to freeze the meme

# Map emotion labels directly to image file paths
MEME_MAP = {
    "asa_mitaka": "asa_mitaka_image.png",
    "shocked": "shocked_image.png",
    "taunt": "taunt_image.png",
    "think_astolfo": "think_astolfo_image.png",
    "thinking_monkey": "thinking_monkey_image.png"
}

# Smoothing buffer: stores the last 10 frame probabilities
prediction_queue = deque(maxlen=10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running inference on: {device}")

# Create named window first so all feeds render inside the same frame
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# -------------------------------------------------------------
# 2. Play Countdown Video
# -------------------------------------------------------------
video_path = os.path.join('..', 'data', '3,2,1 Countdown.mp4')

if os.path.exists(video_path):
    countdown_cap = cv2.VideoCapture(video_path)
    fps = countdown_cap.get(cv2.CAP_PROP_FPS)
    total_frames = countdown_cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cutoff_frame = total_frames - int(fps) if fps > 0 else total_frames
    current_frame = 0

    while countdown_cap.isOpened():
        ret, frame = countdown_cap.read()
        current_frame += 1

        if not ret or current_frame >= cutoff_frame:
            break

        frame_resized = cv2.resize(frame, (PANEL_SIZE[0] * 2, PANEL_SIZE[1]))
        cv2.imshow(WINDOW_NAME, frame_resized)

        delay = int(1000 / fps) if fps > 0 else 30
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    countdown_cap.release()

# -------------------------------------------------------------
# 3. Load Trained PyTorch Model
# -------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Could not find '{MODEL_PATH}' in your project directory!")

checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint['class_names']
num_classes = len(class_names)

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# -------------------------------------------------------------
# 4. PyTorch Preprocessing Pipeline & Haar Cascade
# -------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

xml_path = "haarcascade_frontalface_default.xml"
if not os.path.exists(xml_path):
    import urllib.request

    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, xml_path)

face_cascade = cv2.CascadeClassifier(xml_path)

# -------------------------------------------------------------
# 5. Connect to DroidCam & Perform Real-Time Inference
# -------------------------------------------------------------
endpoints = [f"http://{PHONE_IP}:4747/video", f"http://{PHONE_IP}:4747/mjpegfeed"]
cap = None

for url in endpoints:
    test_cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if test_cap.isOpened():
        ret, frame = test_cap.read()
        if ret and frame is not None and frame.size > 0:
            cap = test_cap
            print(f"[SUCCESS] Connected to DroidCam!")
            break
    test_cap.release()

if cap is None or not cap.isOpened():
    print(f"\n[FAIL] Could not connect to DroidCam.")
    exit()

print("\nStarting Live Emotion Split-Screen Detector! Press 'q' to exit.\n")

# Default placeholder image & timer trackers
current_meme_img = np.zeros((PANEL_SIZE[1], PANEL_SIZE[0], 3), dtype=np.uint8)
last_match_time = 0.0
frozen_label = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        continue

    current_time = time.time()
    is_frozen = (current_time - last_match_time) < FREEZE_DURATION

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_crop = frame[y:y + h, x:x + w]
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)

        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()

        prediction_queue.append(probabilities)
        avg_probabilities = np.mean(prediction_queue, axis=0)

        predicted_idx = np.argmax(avg_probabilities)
        confidence_score = avg_probabilities[predicted_idx] * 100
        emotion_label = class_names[predicted_idx]

        # -------------------------------------------------------------
        # TIMER CONTROL LOGIC
        # -------------------------------------------------------------
        if not is_frozen:
            if emotion_label in MEME_MAP:
                img_path = MEME_MAP[emotion_label]
                if os.path.exists(img_path):
                    loaded_img = cv2.imread(img_path)
                    current_meme_img = cv2.resize(loaded_img, PANEL_SIZE)

                    last_match_time = time.time()
                    frozen_label = emotion_label
                    prediction_queue.clear()

        active_label = frozen_label if is_frozen else emotion_label
        display_text = f"{active_label.capitalize()}: {confidence_score:.1f}%"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y - 35), (x + w, y), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, display_text, (x + 5, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # -------------------------------------------------------------
    # Build Split Screen Window (Left: CAM | Right: MEME)
    # -------------------------------------------------------------
    cam_panel = cv2.resize(frame, PANEL_SIZE)
    meme_panel = current_meme_img.copy()

    # Left Header: Always "CAM"
    cv2.rectangle(cam_panel, (0, 0), (100, 40), (0, 0, 0), cv2.FILLED)
    cv2.putText(cam_panel, "CAM", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Right Header: Clean, untouched "MEME" title
    cv2.rectangle(meme_panel, (0, 0), (120, 40), (0, 0, 0), cv2.FILLED)
    cv2.putText(meme_panel, "MEME", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Freeze Badge: Rendered separately at the bottom-left of MEME panel
    if is_frozen:
        time_left = FREEZE_DURATION - (current_time - last_match_time)
        freeze_text = f"FROZEN: {time_left:.1f}s"

        # Bottom-left dark badge box
        cv2.rectangle(meme_panel, (10, PANEL_SIZE[1] - 45), (180, PANEL_SIZE[1] - 10), (0, 0, 0), cv2.FILLED)
        cv2.putText(meme_panel, freeze_text, (18, PANEL_SIZE[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Combine split screen panels side-by-side
    combined_view = np.hstack((cam_panel, meme_panel))
    cv2.imshow(WINDOW_NAME, combined_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()