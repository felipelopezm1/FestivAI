import cv2
import numpy as np
import requests
from io import BytesIO
import mediapipe as mp
import time
import HandTrackingModule as htm

API_KEY = "bcdb65fc3537c28dca5c1ddc26e07a84"
TILE_SIZE = 256
ZOOM = 5  
CENTER_X, CENTER_Y = 15, 10  #  coordinates roughly over the UK

# Available weather layers from the weatherapi
LAYER_CODES = ["precipitation_new", "clouds_new", "temp_new", "wind_new", "pressure_new", "snow"]
current_layer_idx = 0

cap = cv2.VideoCapture(0)
detector = htm.handDetector(detectionCon=0.7)

last_gesture_time = 0

def fetch_tile(layer, z, x, y):
    url = f"https://tile.openweathermap.org/map/{layer}/{z}/{x}/{y}.png?appid={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        img_arr = np.asarray(bytearray(response.content), dtype=np.uint8)
        tile_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        return tile_img
    else:
        print(f"Failed to fetch tile: {url}")
        return np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    lmList = detector.findPosition(detector.findHands(frame), draw=False)

    # Gesture triggers weather layer change
    if lmList and time.time() - last_gesture_time > 5:
        current_layer_idx = (current_layer_idx + 1) % len(LAYER_CODES)
        print(f"Switched to layer: {LAYER_CODES[current_layer_idx]}")
        last_gesture_time = time.time()

    # Fetch the weather tile as background
    tile = fetch_tile(LAYER_CODES[current_layer_idx], ZOOM, CENTER_X, CENTER_Y)
    if tile is not None:
        tile_resized = cv2.resize(tile, (frame.shape[1], frame.shape[0]))
    else:
        tile_resized = np.zeros_like(frame)

    # Overlay landmarks on the map directly from the logic on the youtube video but did not work the same
    for lm in lmList:
        cx, cy = lm[1], lm[2]
        cv2.circle(tile_resized, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    cv2.putText(tile_resized, f"Layer: {LAYER_CODES[current_layer_idx]}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Weather Map with Gesture Control", tile_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
