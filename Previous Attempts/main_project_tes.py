import streamlit as st
import folium
import requests
from streamlit_folium import st_folium
from geopy.distance import geodesic
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import torch
import mediapipe as mp
from collections import deque
from sklearn.preprocessing import LabelEncoder

# ✅ Weather Config
API_KEY = "0a90412fb5a549d6b34141436250503" #API Key of
OPENWEATHERMAP_API_KEY = "bcdb65fc3537c28dca5c1ddc26e07a84"

class GestureLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GestureLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h, _ = self.lstm(x)
        out = self.dropout(h[:, -1, :])
        return self.fc(out)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GestureLSTM(30, 256, 5).to(device)
model.load_state_dict(torch.load('Gesture\gesture_lstm_model.pth', map_location=device))
model.eval()

le = LabelEncoder()
le.classes_ = np.load('Gesture\gesture_label_classes.npy', allow_pickle=True)


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

# ✅ Streamlit Session State Initialization
st.set_page_config(page_title="ClimaSign - Gesture Weather Control", layout="wide")
st.title("🌦️ Gesture-Controlled Weather Map")

defaults = {
    "selected_weather": "rain",
    "map_location": (51.5074, -0.1278),
    "city": None,
    "condition_text": "",
    "apply_weather_layers": True,
    "predicted_gesture": None,
    "rerun": False,
    "weather_trigger": False,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ✅ Gesture Video Transformer
class GestureVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.sequence_buffer = deque(maxlen=90)
        self.mp_hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        self.SELECTED_INDICES = [0, 4, 8, 12, 20]
        self.last_detected = None
        self.detection_counter = 0
        self.hold_threshold = 90

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(img_rgb)
        combined_landmarks = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                for idx in self.SELECTED_INDICES:
                    lm = hand_landmarks.landmark[idx]
                    combined_landmarks.extend([lm.x, lm.y, lm.z])

        while len(combined_landmarks) < 30:
            combined_landmarks.extend([0.0, 0.0, 0.0])

        if len(combined_landmarks) == 30:
            self.sequence_buffer.append(combined_landmarks)

        if len(self.sequence_buffer) == 90:
            sample = np.array(self.sequence_buffer, dtype=np.float32)
            sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(sample_tensor)
                pred = torch.argmax(output, dim=1).item()
                gesture_label = le.inverse_transform([pred])[0]

            cv2.putText(img, f"Gesture: {gesture_label}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # ✅ 3-sec hold logic
            if gesture_label == self.last_detected:
                self.detection_counter += 1
            else:
                self.detection_counter = 0
                self.last_detected = gesture_label

            if self.detection_counter >= self.hold_threshold:
                st.session_state.predicted_gesture = gesture_label
                st.session_state.rerun = True
                self.detection_counter = 0

        return img

def set_weather(weather):
    st.session_state.selected_weather = weather
    st.session_state.weather_trigger = True

# ✅ GESTURE --> WEATHER Matching Does not work, tried inmany many different ways
gesture_label = st.session_state.get("predicted_gesture")
if gesture_label:
    print(f"✅ Detected Gesture: {gesture_label}")
    gesture_to_weather = {
        "rainy": "rain",
        "cloudy": "cloudy",
        "sunny": "sunny",
        "hail": "hail",
        "snowy": "snow"
    }
    if gesture_label in gesture_to_weather:
        matched_weather = gesture_to_weather[gesture_label]
        print(f"Changing weather to: {matched_weather}")
        st.session_state.selected_weather = matched_weather
        st.session_state.weather_trigger = True
        st.session_state.predicted_gesture = None
        st.experimental_rerun()

# ✅ Streamlit Layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📷 Gesture Camera Feed")
    webrtc_streamer(
        key="gesture-stream",
        video_processor_factory=GestureVideoTransformer,
        media_stream_constraints={"video": True, "audio": False}
    )

    # ✅ Manual Buttons (Optional)
    st.markdown("### 🎮 Weather Controls")
    for label, weather in [("☔ Rain", "rain"), ("☁️ Cloudy", "cloudy"),
                           ("☀️ Sunny", "sunny"), ("🌨️ Hail", "hail"), ("❄️ Snow", "snow")]:
        if st.button(label):
            set_weather(weather)

    st.markdown(f"**☀️ Current Selected Weather:** `{st.session_state.selected_weather}`")

with col2:
    st.markdown("### 🌍 Weather Map")
    if st.session_state.weather_trigger:
        city, condition, _ = find_closest_matching_city(st.session_state.selected_weather)
        if city:
            st.session_state.city = city
            st.session_state.condition_text = condition
            st.session_state.map_location = (city['lat'], city['lon'])
            st.success(f"✅ {city['name']} - {condition.capitalize()}")
        else:
            st.warning(f"No city matches '{st.session_state.selected_weather}'")
        st.session_state.weather_trigger = False

    m = folium.Map(location=st.session_state.map_location, zoom_start=8)
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
                attr="OpenWeatherMap", name=name, overlay=True, control=True
            ).add_to(m)

    if st.session_state.city:
        folium.Marker(
            location=st.session_state.map_location,
            popup=f"{st.session_state.city['name']} - {st.session_state.condition_text}",
            icon=folium.Icon(color="blue")
        ).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, width=None, height=400, key=f"map_{st.session_state.selected_weather}")

# ✅ Optional rerun trigger (shouldn't be needed but kept for safety)
if st.session_state.get("rerun"):
    st.session_state.rerun = False
    st.experimental_rerun()
