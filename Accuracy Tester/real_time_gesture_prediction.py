import cv2
import numpy as np
import torch
import mediapipe as mp
from collections import deque
from sklearn.preprocessing import LabelEncoder

#Load trained LSTM model class (must match training)
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

#Model and label loading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GestureLSTM(input_size=30, hidden_size=256, num_classes=5).to(device)
model.load_state_dict(torch.load('Gesture\gesture_lstm_model.pth', map_location=device))
model.eval()

le = LabelEncoder()
le.classes_ = np.load('Gesture\gesture_label_classes.npy', allow_pickle=True) #sometimes will have to use the regular path instead of the relative, which sucks

#Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

#Real-time capture settings
cap = cv2.VideoCapture(0)
sequence_buffer = deque(maxlen=90)  # Buffer of 90 frames
SELECTED_INDICES = [0, 4, 8, 12, 20]  # Thumb, Index, Middle, Ring, Pinky tips based on https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
FIXED_FEATURES_PER_FRAME = 30  # 5 landmarks * 3D * 2 hands max

print("Real-time Gesture Prediction Started. Press 'q' to exit.")

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

    # Pad if fewer than expected features
    while len(combined_landmarks) < FIXED_FEATURES_PER_FRAME:
        combined_landmarks.extend([0.0, 0.0, 0.0])

    sequence_buffer.append(combined_landmarks)

    # Sequence buffer, help of chatgpt
    if len(sequence_buffer) == 90:
        sample = np.array(sequence_buffer, dtype=np.float32)
        sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(sample_tensor)
            pred = torch.argmax(output, dim=1).item()
            gesture_label = le.inverse_transform([pred])[0]

        cv2.putText(frame, f"Gesture: {gesture_label}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Trigger Print
        if gesture_label == "rainy":
            print("🌧️ Trigger Rain Mode")
        elif gesture_label == "sunny":
            print("☀️ Trigger Sunny Mode")
        elif gesture_label == "hail":
            print("🌨️ Trigger Hail Mode")
        elif gesture_label == "cloudy":
            print("☁️ Trigger Cloudy Mode")
        elif gesture_label == "snowy":
            print("❄️ Trigger Snow Mode")

    cv2.imshow("Real-Time Gesture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
