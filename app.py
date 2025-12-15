from flask import Flask, render_template, request, jsonify
import whisper
import librosa
import numpy as np
import pickle
import tempfile
import subprocess
import os

app = Flask(__name__)

# Load Whisper (tiny = fastest & safest on Render)
whisper_model = whisper.load_model("tiny")

# Load gender model
with open("gender_model.pkl", "rb") as f:
    gender_model = pickle.load(f)

def convert_to_wav(input_path, output_path):
    subprocess.run([
        "ffmpeg", "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def extract_mfcc(wav_path):
    audio, sr = librosa.load(wav_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0).reshape(1, -1)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"text": "No audio", "gender": "Undefined"})

    audio_file = request.files["audio"]

    # Save raw browser audio
    raw_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    audio_file.save(raw_audio.name)

    # Convert to WAV
    wav_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    convert_to_wav(raw_audio.name, wav_audio.name)

    try:
        # Transcription
        result = whisper_model.transcribe(wav_audio.name)
        text = result["text"].strip()

        # Gender prediction
        mfcc = extract_mfcc(wav_audio.name)
        gender = gender_model.predict(mfcc)[0]

    except Exception as e:
        print("ERROR:", e)
        text = "Error in transcription"
        gender = "Undefined"

    finally:
        os.remove(raw_audio.name)
        os.remove(wav_audio.name)

    return jsonify({
        "text": text,
        "gender": gender
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
