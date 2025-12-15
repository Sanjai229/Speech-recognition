import os
import librosa
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X = []
y = []

BASE_DIR = "dataset"
LABELS = ["male", "female"]

print("Loading dataset...")

for label in LABELS:
    folder = os.path.join(BASE_DIR, label)

    for file in os.listdir(folder):
        if not file.lower().endswith(".wav"):
            continue

        path = os.path.join(folder, file)

        try:
            audio, sr = librosa.load(path, sr=16000)

            # Skip very short audio
            if len(audio) < sr:
                continue

            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=13
            )

            mfcc_mean = np.mean(mfcc.T, axis=0)
            X.append(mfcc_mean)
            y.append(label)

        except Exception as e:
            print("Skipped:", file, e)

print("Samples:", len(X))

# ---------- MODEL PIPELINE ----------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

model.fit(X, y)

with open("gender_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("gender_model.pkl saved successfully")
