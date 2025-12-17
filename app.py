from flask import Flask, request, jsonify, render_template
import whisper
import librosa
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load Whisper
print("Loading Whisper tiny model...")
whisper_model = whisper.load_model("tiny")
print("Whisper loaded")

# Load gender model
print("Loading gender model...")
with open("gender_model.pkl", "rb") as f:
    gender_model = pickle.load(f)
print("Gender model loaded")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files["audio"]
    audio_path = os.path.join(UPLOAD_FOLDER, "input.wav")
    audio_file.save(audio_path)

    # -------- TRANSCRIPTION --------
    result = whisper_model.transcribe(audio_path)
    text = result.get("text", "").strip()

    # -------- GENDER PREDICTION --------
    y, sr = librosa.load(audio_path, sr=16000)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=15
    )

    features = np.mean(mfcc.T, axis=0).reshape(1, -1)
    gender = gender_model.predict(features)[0]

    return jsonify({
        "transcription": text if text else "No speech detected",
        "gender": gender
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
