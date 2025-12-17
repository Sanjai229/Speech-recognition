import librosa
import numpy as np
import pickle

# Load trained model
with open("gender_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load audio
y, sr = librosa.load("test.wav", sr=16000)

# IMPORTANT: n_mfcc MUST match training
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=15)
features = np.mean(mfcc.T, axis=0).reshape(1, -1)

# Predict
prediction = model.predict(features)

print("Predicted gender:", prediction[0])
