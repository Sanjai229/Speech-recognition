import os
import subprocess
import whisper
import librosa
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

print("Loading Whisper tiny model...")
whisper_model = whisper.load_model("tiny")
print("Whisper loaded")

print("Loading gender model...")
with open("gender_model.pkl", "rb") as f:
    gender_model = pickle.load(f)
print("Gender model loaded")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        print("Received /transcribe request")

        if "audio" not in request.files:
            print("ERROR: No audio field in request")
            return jsonify({"text": "No audio received", "gender": "Undefined"})

        audio = request.files["audio"]
        print("Audio filename:", audio.filename)

        input_path = os.path.join(UPLOAD_DIR, "input.webm")
        wav_path = os.path.join(UPLOAD_DIR, "audio.wav")

        audio.save(input_path)
        print("Saved input.webm")

        # Convert to WAV
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", wav_path],
            check=True
        )
        print("Converted to WAV")

        # Whisper
        result = whisper_model.transcribe(wav_path, fp16=False)
        text = result["text"].strip()
        print("Whisper text:", text)

        # Gender
        y, sr = librosa.load(wav_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features = np.mean(mfcc.T, axis=0).reshape(1, -1)

        gender = gender_model.predict(features)[0]
        print("Predicted gender:", gender)

        return jsonify({
            "text": text if text else "No speech detected",
            "gender": gender.capitalize()
        })

    except Exception as e:
        print("FATAL ERROR:", e)
        return jsonify({
            "text": "Error in transcription",
            "gender": "Undefined"
        })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
