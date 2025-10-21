import streamlit as st
import folium
import requests
from streamlit_folium import st_folium
from geopy.distance import geodesic
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque, Counter
from sklearn.preprocessing import LabelEncoder
import HandTrackingModule as htm
import time

# API Keys
API_KEY = "0a90412fb5a549d6b34141436250503"
OPENWEATHERMAP_API_KEY = "bcdb65fc3537c28dca5c1ddc26e07a84"

# Initial State Setup
if 'selected_weather' not in st.session_state:
    st.session_state.selected_weather = "rain"
if 'weather_trigger' not in st.session_state:
    st.session_state.weather_trigger = True
if 'photo_processed' not in st.session_state:
    st.session_state.photo_processed = False
if 'city' not in st.session_state:
    st.session_state.city = None
if 'condition_text' not in st.session_state:
    st.session_state.condition_text = ""
if 'map_location' not in st.session_state:
    st.session_state.map_location = (51.5074, -0.1278) #centering london
if 'apply_weather_layers' not in st.session_state:
    st.session_state.apply_weather_layers = True

# UK Cities List
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

# Helper Functions
def get_weather_condition(city_name):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city_name}"
    response = requests.get(url).json()
    return response['current']['condition']['text'].lower()

def find_closest_matching_city(condition):
    matches = []
    for city in UK_CITIES:
        weather = get_weather_condition(city['name'])
        if condition in weather:
            distance = geodesic((51.5074, -0.1278), (city['lat'], city['lon'])).kilometers
            matches.append((city, weather, distance))
    return sorted(matches, key=lambda x: x[2])[0] if matches else (None, None, None)

# Load LSTM Gesture Model
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

model = GestureLSTM(input_size=30, hidden_size=256, num_classes=5)
model.load_state_dict(torch.load(r'C:\Users\felip\Desktop\UAL\MSc\AI For Media\Main Project\AI-4-Media-Project-FelipeLopez\Gesture\gesture_lstm_model.pth', map_location=torch.device('cpu')))
model.eval()

le = LabelEncoder()
le.classes_ = np.load(r'C:\Users\felip\Desktop\UAL\MSc\AI For Media\Main Project\AI-4-Media-Project-FelipeLopez\Gesture\gesture_label_classes.npy', allow_pickle=True)

# Streamlit UI
st.set_page_config(page_title="ClimaSign - Test File", layout="wide")
st.title("Climate and Sign Language Deducer")
st.markdown("#### How's the Weather?")

st.title("✋ Hand Tracking and 🌍 Weather Map Running Together")

col1, col2 = st.columns(2)

### ------------(col1) ---------------- ###
class HandGestureModel(VideoTransformerBase):
    def __init__(self):
        self.sequence_buffer = deque(maxlen=90)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        self.SELECTED_INDICES = [0, 4, 8, 12, 20]
        self.prediction_window = deque(maxlen=45)  # Store last predictions
        self.last_update_time = time.time()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        combined_landmarks = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                for idx in self.SELECTED_INDICES:
                    lm = hand_landmarks.landmark[idx]
                    combined_landmarks.extend([lm.x, lm.y, lm.z])

        while len(combined_landmarks) < 30:
            combined_landmarks.extend([0.0, 0.0, 0.0])

        if len(combined_landmarks) == 30:
            self.sequence_buffer.append(combined_landmarks)

        if len(self.sequence_buffer) == 90:
            sample = np.array(self.sequence_buffer, dtype=np.float32)
            sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                output = model(sample_tensor)
                pred = torch.argmax(output, dim=1).item()
                gesture_label = le.inverse_transform([pred])[0]
                self.prediction_window.append(gesture_label)

            # Check if the same gesture dominates for ~3 seconds (90 frames)
        if len(self.prediction_window) == 45:
            most_common_gesture, count = Counter(self.prediction_window).most_common(1)[0]
            print("Detected:", most_common_gesture, "Count:", count)
            if count > 30 and (time.time() - self.last_update_time) > 2:
                
                
                
                st.session_state.selected_weather = most_common_gesture
                st.session_state.weather_trigger = True
                st.session_state.last_detected_gesture = most_common_gesture  #  Store it
                self.last_update_time = time.time()
                
                if 'last_detected_gesture' in st.session_state: #
                    cv2.putText(img, f"Detected: {st.session_state.last_detected_gesture}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, f"Gesture Triggered: {most_common_gesture}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

with col1:
    st.markdown("### Hand Detection Camera Feed")
    webrtc_streamer(key="hand-detection", video_processor_factory=HandGestureModel, media_stream_constraints={"video": True, "audio": False})

### --------------col 2 ---------------- ###
with col2:
    st.markdown("##### 🌍 Weather Map")

    if 'weather_message' not in st.session_state:
        st.session_state.weather_message = ""

    if st.session_state.weather_trigger:
        city, condition, _ = find_closest_matching_city(st.session_state.selected_weather)
        if city:
            st.session_state.city = city
            st.session_state.condition_text = condition
            st.session_state.map_location = (city['lat'], city['lon'])
            st.session_state.weather_message = f"Closest match: {city['name']} ({city['country']}) - {condition.capitalize()}"
            st.session_state.apply_weather_layers = True
        else:
            st.session_state.city = None
            st.session_state.condition_text = ""
            st.session_state.map_location = (51.5074, -0.1278)
            st.session_state.weather_message = f"No UK cities currently match '{st.session_state.selected_weather}'."
            st.session_state.apply_weather_layers = False

        st.session_state.weather_trigger = False

    if st.session_state.weather_message:
        if st.session_state.city:
            st.success(st.session_state.weather_message)
        else:
            st.warning(st.session_state.weather_message)

    # Draw map
    m = folium.Map(location=st.session_state.map_location, zoom_start=10)

    if st.session_state.apply_weather_layers:
        layers = {
            "Precipitation": "precipitation_new",
            "Clouds": "clouds_new",
            "Temperature": "temp_new",
            "Wind": "wind_new",
            "Pressure": "pressure_new",
            "Snow Depth": "snow"
        }
        for name, code in layers.items():
            folium.raster_layers.TileLayer(
                tiles=f"https://tile.openweathermap.org/map/{code}/{{z}}/{{x}}/{{y}}.png?appid={OPENWEATHERMAP_API_KEY}",
                attr="OpenWeatherMap",
                name=name,
                overlay=True,
                control=True
            ).add_to(m)

    if st.session_state.city:
        folium.Marker(
            location=st.session_state.map_location,
            popup=f"{st.session_state.city['name']}: {st.session_state.condition_text}",
            icon=folium.Icon(color="blue")
        ).add_to(m)

    folium.LayerControl().add_to(m)

    st_folium(m, width=None, height=400, key=f"map_{st.session_state.selected_weather}")
