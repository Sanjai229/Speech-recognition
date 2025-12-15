# train_gender_model.py
import librosa
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import pickle

X = []
y = []

for label in ["male", "female"]:
    folder = f"dataset/{label}"
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)
            y_audio, sr = librosa.load(path, sr=16000)
            mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
            X.append(np.mean(mfcc.T, axis=0))
            y.append(label)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, y)

with open("gender_model.pkl", "wb") as f:
    pickle.dump(clf, f)
