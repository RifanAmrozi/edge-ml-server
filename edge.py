import cv2
import requests
import time
import datetime

# TODO: replace with actual RTSP URL of the camera
# RTSP_URL = "rtsp://<camera_ip>/stream"
RTSP_URL = "rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa"  # example public stream
API_URL = "http://localhost:8080/api/v1/save"  # later replace with Go server endpoint

cap = cv2.VideoCapture(RTSP_URL)
if not cap.isOpened():
    print("❌ Cannot open RTSP stream")
    exit()

frame_count = 0
start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame, stopping...")
        break

    # TODO: process frame (e.g., object detection, machine learning inference, etc.)
    frame_count += 1
    if frame_count % 60 == 0:  # every ~60 frames
        fps = frame_count / (time.time() - start)
        payload = {
            "camera_id": "cam-01",
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "fps": fps,
            "frame_count": frame_count
        }
        r = requests.post(API_URL, json=payload)
        print("POST result:", r.status_code)
