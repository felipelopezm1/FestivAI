import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GestureDataset(Dataset):
    def __init__(self, X, y, fixed_length=90):
        
        cleaned_X, valid_y = [], []
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


class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GestureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)  # ✅ Match num_layers
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h, _ = self.lstm(x)
        out = self.dropout(h[:, -1, :])
        out = self.fc(out)
        return out


X = np.load('X_sequences.npy', allow_pickle=True)
y = np.load('y_labels.npy', allow_pickle=True)

#Load model with the exact same hidden size and layers as training (Help of ChatGPT and week on training)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)


dataset = GestureDataset(X, y_encoded)
data_loader = DataLoader(dataset, batch_size=4, shuffle=False)


model = GestureLSTM(input_size=30, hidden_size=256, num_classes=num_classes).to(device)
model.load_state_dict(torch.load('Gesture\gesture_lstm_model.pth'))
model.eval()

#Inference loop
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in data_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

#Report
print("\n✅ Accuracy:", accuracy_score(all_labels, all_preds) * 100, "%")
print("\n✅ Classification Report:")
print(classification_report(
    all_labels,
    all_preds,
    labels=range(num_classes),
    target_names=le.classes_,
    zero_division=0
))
