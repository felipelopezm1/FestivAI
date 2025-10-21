import streamlit as st
import folium
import requests
from streamlit_folium import st_folium
from geopy.distance import geodesic
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

# API Keys
API_KEY = "0a90412fb5a549d6b34141436250503" #API Key of
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
    st.session_state.map_location = (51.5074, -0.1278)
if 'apply_weather_layers' not in st.session_state:
    st.session_state.apply_weather_layers = True  # Whether to show weather layers at all

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

# Streamlit UI
st.set_page_config(page_title="ClimaSign - Test File", layout="wide")
st.title("Climate and Sign Language Deducer")
st.markdown("#### How's the Weather?")

class HandDetector(VideoTransformerBase):
    def __init__(self):
        self.detector = htm.handDetector()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = self.detector.findHands(img)
        lmList = self.detector.findPosition(img)

        # Store hand positions for later control logic
        st.session_state['lmList'] = lmList

        if len(lmList) != 0:
            cv2.putText(img, f"Landmark 4: {lmList[4]}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #set_weather("rain")

        return img


st.title("✋ Hand Tracking and 🌍 Weather Map Running Together")


col1, col2 = st.columns(2)

### ---------------- CAMERA STREAM (col1) ---------------- ###
with col1:
    st.markdown("### Hand Detection Camera Feed")
    webrtc_streamer(key="hand-detection", video_processor_factory=HandDetector)
    if webrtc_streamer == True :
        print('works')

### ---------------- WEATHER AND MAP (col2) ---------------- ###
with col2:
    st.markdown("##### 🌍 Weather Map")

    cols = st.columns(5)

    def set_weather(weather):
        st.session_state.selected_weather = weather
        st.session_state.weather_trigger = True

    if cols[0].button("☔ Rain"): set_weather("rain")
    if cols[1].button("☁️ Cloudy"): set_weather("cloudy")
    if cols[2].button("☀️ Sunny"): set_weather("sunny")
    if cols[3].button("🌨️ Hail"): set_weather("hail")
    if cols[4].button("❄️ Snow"): set_weather("snow")

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

    # Show last weather message
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

    st.markdown(
        "<style>#map-container {opacity: 0.85;}</style>",
        unsafe_allow_html=True
    )

    st_folium(m, width=None, height=400, key=f"map_{st.session_state.selected_weather}")