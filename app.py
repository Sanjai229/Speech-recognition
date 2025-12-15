from flask import Flask, render_template, request, jsonify
import whisper
import librosa
import numpy as np
import pickle
import tempfile
import os

app = Flask(__name__)

# Load Whisper tiny (fast & cheap for Render)
whisper_model = whisper.load_model("tiny")

# Load trained gender model
with open("gender_model.pkl", "rb") as f:
    gender_model = pickle.load(f)

def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0).reshape(1, -1)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"})

    audio_file = request.files["audio"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio_file.save(tmp.name)
        audio_path = tmp.name

    # Speech to text
    result = whisper_model.transcribe(audio_path)
    text = result["text"]

    # Gender prediction
    mfcc = extract_mfcc(audio_path)
    gender = gender_model.predict(mfcc)[0]

    os.remove(audio_path)

    return jsonify({
        "text": text,
        "gender": gender
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
