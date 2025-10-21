#Rain and Hail https://www.youtube.com/watch?v=Ydn4WQ8HVIg
#Cloudy https://www.youtube.com/watch?v=Ydn4WQ8HVIg
#Snowy https://www.youtube.com/watch?v=iyMAMfPJC54
#Sunny https://www.youtube.com/watch?v=OVtfooojJD4

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, accuracy_score

#  Hyperparameters
EPOCHS = 400
BATCH_SIZE = 4
HIDDEN_SIZE = 256       
LEARNING_RATE = 0.001
FIXED_SEQUENCE_LENGTH = 90
FEATURE_SIZE = 30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#  Custom Dataset with cleaning and padding
class GestureDataset(Dataset):
    def __init__(self, X, y, fixed_length=90):
        cleaned_X = []
        valid_y = []
        for i, seq in enumerate(X):
            cleaned_seq = []
            for frame in seq:
                frame = np.array(frame)
                if frame.shape[0] == 30:
                    cleaned_seq.append(frame)
            if not cleaned_seq:
                continue
            cleaned_seq = np.array(cleaned_seq, dtype=np.float32)
            if len(cleaned_seq) < fixed_length:
                pad_amount = fixed_length - len(cleaned_seq)
                padding = np.zeros((pad_amount, 30), dtype=np.float32)
                cleaned_seq = np.vstack([cleaned_seq, padding])
            elif len(cleaned_seq) > fixed_length:
                cleaned_seq = cleaned_seq[:fixed_length]
            cleaned_X.append(cleaned_seq)
            valid_y.append(y[i])
        self.X = torch.tensor(np.array(cleaned_X), dtype=torch.float32)
        self.y = torch.tensor(np.array(valid_y), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#LSTM Model
class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GestureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h, _ = self.lstm(x)
        out = h[:, -1, :]
        out = self.fc(out)
        return out

#Load your new dataset
X = np.load('X_sequences.npy', allow_pickle=True)
y = np.load('y_labels.npy', allow_pickle=True)

#Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

#Build dataset and DataLoader
dataset = GestureDataset(X, y_encoded, fixed_length=FIXED_SEQUENCE_LENGTH)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

#Initialize model
model = GestureLSTM(FEATURE_SIZE, HIDDEN_SIZE, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

#Optional Evaluation from here downward a lot of help from ChatGPT and the training week
print(" Starting LSTM training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss:.4f}")

#Save model and label classes
torch.save(model.state_dict(), "gesture_lstm_model.pth")
np.save('gesture_label_classes.npy', le.classes_)
print("LSTM model and label encoder saved!")

#Optional Evaluation
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

print("\n Evaluation Accuracy:", accuracy_score(all_labels, all_preds) * 100, "%")
print("\n Classification Report:")
print(classification_report(all_labels, all_preds, labels=range(num_classes), target_names=le.classes_))

print(classification_report( 
    all_labels, 
    all_preds, 
    labels=range(num_classes),   #  Force all label indices
    target_names=le.classes_     #  Keep your gesture names
))
