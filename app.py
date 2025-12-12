from flask import Flask, render_template, request, jsonify
import whisper
import torch
import soundfile as sf
from pyannote.audio import Model
import os
import numpy as np

app = Flask(__name__)

# ------------------------
# Dynamic Uploads Folder
# ------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------------
# Load Models
# ------------------------
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper loaded.")

print("Loading Pyannote gender model...")
gender_model = Model.from_pretrained("pyannote/gender", use_auth_token=None)
print("Gender model loaded.")

# ------------------------
# Routes
# ------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "audio_data" not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400

    audio_file = request.files["audio_data"]
    filename = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(filename)

    # --- 1) Transcription ---
    result = whisper_model.transcribe(filename)
    transcription = result["text"]

    # --- 2) Gender Prediction ---
    waveform, sample_rate = sf.read(filename)
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)  # mono
    waveform_tensor = torch.tensor(waveform).float().unsqueeze(0)

    with torch.no_grad():
        gender_result = gender_model(waveform_tensor, sample_rate=sample_rate)

    male_score = float(gender_result["male"])
    female_score = float(gender_result["female"])
    if male_score > female_score:
        gender_label = "Male"
        confidence = male_score
    else:
        gender_label = "Female"
        confidence = female_score

    return jsonify({
        "transcription": transcription,
        "gender": gender_label,
        "confidence": round(confidence*100, 2)
    })


# ------------------------
# Run server (Render-ready)
# ------------------------
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
