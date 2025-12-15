import os
import tempfile
import subprocess
import pickle

import numpy as np
import librosa
from flask import Flask, request, jsonify, render_template

import whisper

# ===============================
# Flask App
# ===============================
app = Flask(__name__)

# ===============================
# Load Whisper (TINY model)
# ===============================
print("Loading Whisper tiny model...")
whisper_model = whisper.load_model("tiny")  # VERY IMPORTANT for Render memory
print("Whisper loaded")

# ===============================
# Load Gender Model
# ===============================
print("Loading gender model...")
with open("gender_model.pkl", "rb") as f:
    gender_model = pickle.load(f)
print("Gender model loaded")

# ===============================
# Audio Utilities
# ===============================
def convert_to_wav(input_path, output_path):
    """
    Convert webm/opus to wav using ffmpeg
    """
    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        output_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def extract_mfcc(wav_path):
    """
    Extract MFCC features for gender detection
    """
    audio, sr = librosa.load(wav_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean.reshape(1, -1)

# ===============================
# Routes
# ===============================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({
            "text": "No audio received",
            "gender": "Undefined"
        })

    audio_file = request.files["audio"]
    print("Received:", audio_file.filename)

    # Save raw audio
    raw_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    audio_file.save(raw_audio.name)

    # Convert to wav
    wav_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    convert_to_wav(raw_audio.name, wav_audio.name)

    try:
        # ===== Whisper transcription =====
        result = whisper_model.transcribe(
            wav_audio.name,
            fp16=False   # CPU safe
        )
        text = result["text"].strip()

        # ===== Gender detection =====
        mfcc = extract_mfcc(wav_audio.name)
        gender = gender_model.predict(mfcc)[0]

    except Exception as e:
        print("ERROR:", e)
        text = "Error in transcription"
        gender = "Undefined"

    # Cleanup temp files
    os.remove(raw_audio.name)
    os.remove(wav_audio.name)

    return jsonify({
        "text": text,
        "gender": gender
    })


# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
