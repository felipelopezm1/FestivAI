import cv2
import mediapipe as mp
import numpy as np
import time
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Storage
sequences = []
labels = []

output_X = 'X_sequences.npy'
output_y = 'y_labels.npy'

# Parameters
SAMPLES_PER_GESTURE = 10
RECORD_SECONDS = 5 #found that this many seconds gave me enough time to do the whole movement
FIXED_FEATURES_PER_FRAME = 30
SELECTED_INDICES = [0, 4, 8, 12, 20]  

cap = cv2.VideoCapture(0)

gesture_classes = ['cloudy', 'hail', 'rainy', 'snowy', 'sunny'] 

print(" 10 samples Press 's' to start each gesture recording.")

for gesture in gesture_classes:
    print(f"Ready to collect samples for gesture: **{gesture}**")

    collected = 0
    while collected < SAMPLES_PER_GESTURE:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.putText(frame, f"Gesture: {gesture} | Press 's' to RECORD | Collected: {collected}/{SAMPLES_PER_GESTURE}", #based on training week we had
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Dataset Collector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print(f"🎬 Recording sample {collected + 1} for {gesture}...")
            sequence = []
            start_time = time.time()
            while time.time() - start_time < RECORD_SECONDS:
                ret, frame = cap.read()
                if not ret:
                    break


                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                combined_landmarks = []

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) #based on project 1 video https://www.computervision.zone/topic/volumehandcontrol-py/
                        for idx in SELECTED_INDICES:
                            lm = hand_landmarks.landmark[idx]
                            combined_landmarks.extend([lm.x, lm.y, lm.z])

                # Pad if only one hand or missing
                while len(combined_landmarks) < FIXED_FEATURES_PER_FRAME: #help of chatgpt to get the frames normalized, was getting a lot of errors
                    combined_landmarks.extend([0.0, 0.0, 0.0])

                sequence.append(combined_landmarks)
                cv2.imshow("Dataset Collector", frame)
                cv2.waitKey(1)

            sequences.append(sequence)
            labels.append(gesture)
            collected += 1
            print(f" Sample {collected}/{SAMPLES_PER_GESTURE} for {gesture} recorded!")

        if key == ord('q'):
            print(" Exiting early...")
            break

print(" Dataset collection complete. Saving...")

cap.release()
cv2.destroyAllWindows()

np.save(output_X, np.array(sequences, dtype=object), allow_pickle=True)
np.save(output_y, np.array(labels), allow_pickle=True)

print(f" Dataset saved as '{output_X}' and '{output_y}'. Total samples: {len(sequences)}")
