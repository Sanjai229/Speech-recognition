import os
import pickle
import numpy as np
import librosa
import soundfile
import whisper
from flask import Flask, render_template, request, jsonify

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
            return jsonify({"text": "No audio received", "gender": "Unknown"})

        audio = request.files["audio"]

        webm_path = os.path.join(UPLOAD_DIR, "audio.webm")
        wav_path = os.path.join(UPLOAD_DIR, "audio.wav")

        audio.save(webm_path)
        print("Saved webm file")

        # ✅ SAFE DECODE (NO FFMPEG)
        y, sr = librosa.load(webm_path, sr=16000)

        if y is None or len(y) < 1000:
            print("Audio empty or too short")
            return jsonify({"text": "No speech detected", "gender": "Unknown"})

        soundfile.write(wav_path, y, sr)
        print("WAV written. Samples:", len(y))

        # 🔹 WHISPER TRANSCRIPTION
        result = whisper_model.transcribe(wav_path, fp16=False)
        text = result.get("text", "").strip()
        print("Whisper output:", text)

        # 🔹 GENDER PREDICTION
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features = np.mean(mfcc.T, axis=0).reshape(1, -1)

        gender = gender_model.predict(features)[0]
        print("Predicted gender:", gender)

        return jsonify({
            "text": text if text else "No speech detected",
            "gender": gender.capitalize()
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({
            "text": "Error in transcription",
            "gender": "Unknown"
        })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
