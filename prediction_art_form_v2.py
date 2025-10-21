import cv2
import numpy as np
import torch
import mediapipe as mp
from collections import deque
from sklearn.preprocessing import LabelEncoder
import requests
from geopy.distance import geodesic
import time
import os
from PIL import Image
from stable_diffusion_pytorch import pipeline, model_loader

#Weather API Key
API_KEY = "0a90412fb5a549d6b34141436250503"

#UK Cities Batch
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

#Weather fetching function
def get_weather_condition(city_name):
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city_name}"
        response = requests.get(url).json()
        return response['current']['condition']['text'].lower()
    
    except:
        return "Not found"

def find_closest_matching_city(condition):
    global weather_cache, last_fetch_time
    if time.time() - last_fetch_time > CACHE_INTERVAL:
        update_weather_cache()
        
    matches = []
    for city in UK_CITIES:
        weather = weather_cache.get(city['name'], "unknown")
        if condition in weather:
            distance = geodesic((51.5074, -0.1278), (city['lat'], city['lon'])).kilometers #calculate lat and lon, got help from chatgpt in this section
            matches.append((city, weather, distance))
    return sorted(matches, key=lambda x: x[2])[0] if matches else (None, None, None)

#Stable Diffusion Model load (Only ONCE)
print("Loading Stable Diffusion models")
models = model_loader.preload_models('cpu')

print("Models loaded")

#LSTM Model Class
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

#Model loading 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #for when I had to work on it onmy nvidia pc
model = GestureLSTM(input_size=30, hidden_size=256, num_classes=5).to(device)
model.load_state_dict(torch.load('Gesture/gesture_lstm_model.pth', map_location=device))
model.eval()

le = LabelEncoder()
le.classes_ = np.load('Gesture/gesture_label_classes.npy', allow_pickle=True)

#MediaPipe hands init
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

#Webcam init
cap = cv2.VideoCapture(0) #
sequence_buffer = deque(maxlen=30)
SELECTED_INDICES = [0, 4, 8, 12, 20]
FIXED_FEATURES_PER_FRAME = 30

#Weather cache setup in order for the model to not get too slow
current_city = None
current_condition = None
weather_cache = {}
last_fetch_time = 0
CACHE_INTERVAL = 600  #double the cache

def update_weather_cache():
    global weather_cache, last_fetch_time
    weather_cache = {}
    for city in UK_CITIES:
        condition = get_weather_condition(city['name'])
        weather_cache[city['name']] = condition
    last_fetch_time = time.time()

update_weather_cache()

print(" Real-time Gesture Weather Generator Started. Press 'q' to exit.")

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

    if len(sequence_buffer) >= 30: #changed it to more than 90 as it was lagging a lot
        sample = np.array(sequence_buffer, dtype=np.float32)
        sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(sample_tensor)
            pred = torch.argmax(output, dim=1).item()
            gesture_label = le.inverse_transform([pred])[0]

        #Display gesture
        cv2.putText(frame, f"Gesture: {gesture_label}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        #If gesture matches, find city and generate image
        if gesture_label in ["rainy", "sunny", "hail", "cloudy", "snowy"]:
            city, condition, _ = find_closest_matching_city(gesture_label)
            if city:
                current_city = city
                current_condition = condition

                #table Diffusion image generation, got a lot of help from ChatGPT in order to implement the weekb 6a integration with the prediction_art_form v1
                prompt = f"A {gesture_label} day in {city['name']} painted in the impressionist style of Claude Monet. The scene is rendered with soft, harmonious colors and expressive, textured brushstrokes. Light and shadow blend seamlessly, creating a dreamy, atmospheric quality. Buildings, rivers, streets, and landscapes appear as blurred, delicate forms immersed in layers of misty tones. The overall composition feels serene, fluid, and painterly, evoking the timeless beauty of Monet’s masterpieces."
                prompts = [prompt]
                print(f"Generating image for: {prompt}")

                image = pipeline.generate( 
                    prompts=prompts,
                    uncond_prompts=None,
                    input_images=[],
                    strength=0.8,
                    do_cfg=True,
                    cfg_scale=7.5,
                    height=512,
                    width=512,
                    sampler="k_lms",
                    n_inference_steps=20,
                    seed=None,
                    models=models,
                    device='cpu',
                    idle_device='cpu'
                )[0]

                image_pil = Image.fromarray(np.asarray(image))
                output_folder = "generated_images"
                os.makedirs(output_folder, exist_ok=True)
                filename = os.path.join(output_folder, f"{gesture_label}_{city['name']}.png")
                image_pil.save(filename)
                print(f" Generated image saved at: {filename}")
            else:
                current_city = None
                current_condition = None

    #Overlay closest city and weather
    if current_city:
        cv2.putText(frame, f"Closest City: {current_city['name']} ({current_city['country']})",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Condition: {current_condition.capitalize()}",
                    (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    else:
        cv2.putText(frame, "No matching city found.",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    #  Show video feed Generation in 
    cv2.imshow("Gesture Controlled Weather Generator", frame)

    #  Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  Clean up
cap.release()
cv2.destroyAllWindows()
