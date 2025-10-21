import cv2
import numpy as np
import torch
import requests
import time
from collections import deque
from sklearn.preprocessing import LabelEncoder
import mediapipe as mp

#  OpenWeatherMap API Config
OPENWEATHERMAP_API_KEY = "bcdb65fc3537c28dca5c1ddc26e07a84"
TILE_SIZE = 256
ZOOM = 5
CENTER_X, CENTER_Y = 15, 10  # London area

#  Weather Layers (Streamlit/Folium compatible)
LAYER_NAME_TO_CODE = {
    "Precipitation": "precipitation_new",
    "Clouds": "clouds_new",
    "Temperature": "temp_new",
    "Wind": "wind_new",
    "Pressure": "pressure_new",
    "Snow Depth": "snow"
}

#  Map specific gestures to specific layers
GESTURE_TO_LAYER = {
    "rainy": "Precipitation",
    "cloudy": "Wind",
    "sunny": "Temperature",
    "hail": "Pressure",
    "snowy": "Snow Depth"
}

#  LSTM Model Class
class GestureLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GestureLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h, _ = self.lstm(x)
        out = self.dropout(h[:, -1, :])
        out = self.fc(out)
        return out

#  Load Model and Labels
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GestureLSTM(input_size=30, hidden_size=256, num_classes=5).to(device)
model.load_state_dict(torch.load(r'Gesture\gesture_lstm_model.pth', map_location=device))
model.eval()

le = LabelEncoder()
le.classes_ = np.load(r'Gesture\gesture_label_classes.npy', allow_pickle=True)
print(f" Model loaded. Gesture classes: {le.classes_}")

#  MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

#  Webcam + Buffer
cap = cv2.VideoCapture(0)
sequence_buffer = deque(maxlen=90)
SELECTED_INDICES = [0, 4, 8, 12, 20]
FIXED_FEATURES_PER_FRAME = 30

#  Fetch Weather Tile
def fetch_tile(layer_code, z, x, y):
    url = f"https://tile.openweathermap.org/map/{layer_code}/{z}/{x}/{y}.png?appid={OPENWEATHERMAP_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        img_arr = np.asarray(bytearray(response.content), dtype=np.uint8)
        return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    else:
        print(f"Failed to fetch tile: {url}")
        return np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)

#  Start with a default layer
current_layer_name = "Clouds"
last_switch_time = 0
gesture_cooldown = 5  # seconds between changes

print(" specific Weather Layer Mapping Active. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    combined_landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for idx in SELECTED_INDICES:
                lm = hand_landmarks.landmark[idx]
                combined_landmarks.extend([lm.x, lm.y, lm.z])

    #  Always pad to 30 features
    if len(combined_landmarks) == 0:
        combined_landmarks = [0.0] * FIXED_FEATURES_PER_FRAME
    elif len(combined_landmarks) < FIXED_FEATURES_PER_FRAME:
        combined_landmarks.extend([0.0] * (FIXED_FEATURES_PER_FRAME - len(combined_landmarks)))

    sequence_buffer.append(combined_landmarks)

    #  Predict gesture
    if len(sequence_buffer) == 90:
        sample = np.array(list(sequence_buffer), dtype=np.float32)
        sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(sample_tensor)
            pred = torch.argmax(output, dim=1).item()
            gesture_label = le.inverse_transform([pred])[0]
        
        print(f"Predicted Gesture: {gesture_label}")

        #  Switch only if gesture is mapped and cooldown passed
        if gesture_label in GESTURE_TO_LAYER and time.time() - last_switch_time > gesture_cooldown:
            current_layer_name = GESTURE_TO_LAYER[gesture_label]
            last_switch_time = time.time()
            print(f" Gesture '{gesture_label}' ➔ Switched to Layer '{current_layer_name}'")

    #  Fetch weather tile of current layer
    current_layer_code = LAYER_NAME_TO_CODE[current_layer_name]
    tile = fetch_tile(current_layer_code, ZOOM, CENTER_X, CENTER_Y)
    tile_resized = cv2.resize(tile, (frame.shape[1], frame.shape[0]))
    blended = cv2.addWeighted(tile_resized, 0.7, frame, 0.3, 0)
    cv2.putText(blended, f"Weather Layer: {current_layer_name}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("🌦️ Gesture ➔ Weather Layer Map", blended)

    #  Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
