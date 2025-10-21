import cv2
import numpy as np
import torch
import mediapipe as mp
from collections import deque
from sklearn.preprocessing import LabelEncoder
import requests
from geopy.distance import geodesic
import time

# API Key (WeatherAPI)
API_KEY = "0a90412fb5a549d6b34141436250503"

# ✅ UK Cities Example (add more if needed)
UK_CITIES = [
    {"name": "London", "lat": 51.5074, "lon": -0.1276, "country": "UK"},
    {"name": "Manchester", "lat": 53.4808, "lon": -2.2426, "country": "UK"},
    {"name": "Birmingham", "lat": 52.4862, "lon": -1.8904, "country": "UK"},
    {"name": "Liverpool", "lat": 53.4084, "lon": -2.9916, "country": "UK"},
    {"name": "Leeds", "lat": 53.8008, "lon": -1.5491, "country": "UK"},
    {"name": "Glasgow", "lat": 55.8642, "lon": -4.2518, "country": "UK"},
    {"name": "Edinburgh", "lat": 55.9533, "lon": -3.1883, "country": "UK"},
    {"name": "Bristol", "lat": 51.4545, "lon": -2.5879, "country": "UK"},
    {"name": "Cardiff", "lat": 51.4816, "lon": -3.1791, "country": "UK"},
    {"name": "Belfast", "lat": 54.5973, "lon": -5.9301, "country": "UK"},
    {"name": "Sheffield", "lat": 53.3811, "lon": -1.4701, "country": "UK"},
    {"name": "Newcastle", "lat": 54.9784, "lon": -1.6174, "country": "UK"},
    {"name": "Southampton", "lat": 50.9097, "lon": -1.4044, "country": "UK"},
    {"name": "Nottingham", "lat": 52.9548, "lon": -1.1581, "country": "UK"},
    {"name": "Leicester", "lat": 52.6369, "lon": -1.1398, "country": "UK"},
    {"name": "Oxford", "lat": 51.7520, "lon": -1.2577, "country": "UK"},
    {"name": "Cambridge", "lat": 52.2053, "lon": 0.1218, "country": "UK"},
]

# ✅ Weather fetching function
def get_weather_condition(city_name):
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city_name}"
        response = requests.get(url).json()
        return response['current']['condition']['text'].lower()
    except:
        return "unknown"

# ✅ Find the closest UK city matching the weather condition
def find_closest_matching_city(condition):
    global weather_cache, last_fetch_time
    # Update cache if stale
    if time.time() - last_fetch_time > CACHE_INTERVAL:
        update_weather_cache()

    matches = []
    for city in UK_CITIES:
        weather = weather_cache.get(city['name'], "unknown")
        if condition in weather:
            distance = geodesic((51.5074, -0.1278), (city['lat'], city['lon'])).kilometers
            matches.append((city, weather, distance))
    return sorted(matches, key=lambda x: x[2])[0] if matches else (None, None, None)

# ✅ LSTM Model Class
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

# ✅ Model and label loading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GestureLSTM(input_size=30, hidden_size=256, num_classes=5).to(device)
model.load_state_dict(torch.load('Gesture\gesture_lstm_model.pth', map_location=device))
model.eval()

le = LabelEncoder()
le.classes_ = np.load('Gesture\gesture_label_classes.npy', allow_pickle=True)

# ✅ MediaPipe hands init
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ✅ Webcam init
cap = cv2.VideoCapture(0)
sequence_buffer = deque(maxlen=90)  # Buffer of 90 frames
SELECTED_INDICES = [0, 4, 8, 12, 20]  # Thumb, Index, Middle, Ring, Pinky tips
FIXED_FEATURES_PER_FRAME = 30  # 5 points * 3D * 2 hands max

# ✅ Store current match
current_city = None
current_condition = None

weather_cache = {}
last_fetch_time = 0
CACHE_INTERVAL = 300  # 5 minutes

def update_weather_cache():
    global weather_cache, last_fetch_time
    weather_cache = {}
    for city in UK_CITIES:
        condition = get_weather_condition(city['name'])
        weather_cache[city['name']] = condition
    last_fetch_time = time.time()

# Call this once outside the main loop to prefill cache
update_weather_cache()

print("✅ Real-time Gesture Prediction Started. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    combined_landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for idx in SELECTED_INDICES:
                lm = hand_landmarks.landmark[idx]
                combined_landmarks.extend([lm.x, lm.y, lm.z])

    while len(combined_landmarks) < FIXED_FEATURES_PER_FRAME:
        combined_landmarks.extend([0.0, 0.0, 0.0])

    sequence_buffer.append(combined_landmarks)

    if len(sequence_buffer) == 90:
        sample = np.array(sequence_buffer, dtype=np.float32)
        sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(sample_tensor)
            pred = torch.argmax(output, dim=1).item()
            gesture_label = le.inverse_transform([pred])[0]

        # ✅ Display gesture
        cv2.putText(frame, f"Gesture: {gesture_label}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ✅ If gesture matches, search closest city
        if gesture_label in ["rainy", "sunny", "hail", "cloudy", "snowy"]:
            city, condition, _ = find_closest_matching_city(gesture_label)
            if city:
                current_city = city
                current_condition = condition
            else:
                current_city = None
                current_condition = None

    # ✅ Overlay closest city and weather
    if current_city:
        cv2.putText(frame, f"Closest City: {current_city['name']} ({current_city['country']})",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Condition: {current_condition.capitalize()}",
                    (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    else:
        cv2.putText(frame, "No matching city found.",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # ✅ Show the video feed
    cv2.imshow("Gesture Controlled Weather Info", frame)

    # ✅ Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Clean up
cap.release()
cv2.destroyAllWindows()
