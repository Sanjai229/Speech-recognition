import os
import tempfile
import pickle
import numpy as np
import librosa
import whisper

from flask import Flask, render_template, request, jsonify

# ---------------- CONFIG ----------------
PORT = int(os.environ.get("PORT", 10000))

# ---------------- LOAD MODELS (ONCE) ----------------
print("Loading Whisper tiny model...")
whisper_model = whisper.load_model("tiny")
print("Whisper loaded")

print("Loading gender model...")
with open("gender_model.pkl", "rb") as f:
    gender_model = pickle.load(f)
print("Gender model loaded")

# ---------------- FLASK APP ----------------
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"text": "No audio received", "gender": "Undefined"})

    audio_file = request.files["audio"]

    # Save temp wav
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio_path = tmp.name
        audio_file.save(audio_path)

    try:
        # ---------- TRANSCRIPTION ----------
        result = whisper_model.transcribe(
            audio_path,
            fp16=False
        )
        text = result.get("text", "").strip()

        # ---------- GENDER PREDICTION ----------
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)

        gender = gender_model.predict(mfcc_mean)[0]

        return jsonify({
            "text": text if text else "No speech detected",
            "gender": gender.capitalize()
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({
            "text": "Error in transcription",
            "gender": "Undefined"
        })

    finally:
        os.remove(audio_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
