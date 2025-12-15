import os
import librosa
import numpy as np
import pickle

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

DATASET_PATH = "dataset"

X = []
y = []

def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

for label in ["Male", "Female"]:
    folder = os.path.join(DATASET_PATH, label)
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)
            X.append(extract_mfcc(path))
            y.append(label)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", probability=True)
)

model.fit(X, y)

with open("gender_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Gender model trained & saved as gender_model.pkl")
