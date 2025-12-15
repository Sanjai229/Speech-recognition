import os
import whisper
import librosa
import numpy as np
import pickle
import subprocess
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

print("Loading Whisper tiny model...")
whisper_model = whisper.load_model("tiny")
print("Whisper loaded")

print("Loading gender model...")
with open("gender_model.pkl", "rb") as f:
    gender_model, scaler = pickle.load(f)
print("Gender model loaded")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        if "audio" not in request.files:
            print("❌ No audio file received")
            return jsonify({"text": "No audio received", "gender": "Undefined"})

        audio_file = request.files["audio"]

        if audio_file.filename == "":
            print("❌ Empty filename")
            return jsonify({"text": "Empty audio file", "gender": "Undefined"})

        input_path = "input.webm"
        wav_path = "input.wav"

        audio_file.save(input_path)
        print("✅ Audio saved")

        # Convert to WAV using ffmpeg
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        print("✅ Converted to WAV")

        # Transcription
        result = whisper_model.transcribe(wav_path)
        text = result["text"]

        print("📝 Transcription:", text)

        # Gender detection
        audio, sr = librosa.load(wav_path, sr=16000)
        mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T, axis=0)
        mfcc_scaled = scaler.transform([mfcc])
        gender = gender_model.predict(mfcc_scaled)[0]

        print("👤 Gender:", gender)

        return jsonify({
            "text": text,
            "gender": gender
        })

    except Exception as e:
        print("🔥 ERROR:", str(e))
        return jsonify({"text": "Error in transcription", "gender": "Undefined"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
