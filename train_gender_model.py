import librosa
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier

X = []
y = []

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000)

    # MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Pitch (VERY IMPORTANT)
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    pitch_values = pitches[pitches > 0]

    pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
    pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0

    return np.hstack([mfcc_mean, pitch_mean, pitch_std])

for label in ["male", "female"]:
    folder = f"dataset/{label}"
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)
            features = extract_features(path)
            X.append(features)
            y.append(label)

clf = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

clf.fit(X, y)

with open("gender_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("✅ Gender model trained successfully")
