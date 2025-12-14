from flask import Flask, request, jsonify, render_template
import openai_whisper as whisper
import os
import numpy as np
from pydub import AudioSegment
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

app = Flask(__name__)

# Load Whisper model for transcription
model = whisper.load_model("base")  # Options: "small", "medium", "large"

# Dummy gender detection (replace with your trained model or simple rules)
# Here we just simulate a gender classifier for demonstration
# In practice, you'd train an SVM or neural network with audio features

# Example SVM model (dummy, random weights)
scaler = StandardScaler()
gender_model = SVC(probability=True)

def detect_gender(audio_path):
    # Convert to mono and 16kHz
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    
    # Dummy feature extraction: use mean and std
    features = np.array([samples.mean(), samples.std()]).reshape(1, -1)
    features = scaler.fit_transform(features)  # scale
    
    # Dummy prediction
    pred = gender_model.fit(features, [0]).predict(features)  # always returns 0 (male)
    return "Male" if pred[0] == 0 else "Female"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    
    audio_file = request.files["audio"]
    audio_path = "temp.wav"
    audio_file.save(audio_path)

    # Transcription
    result = model.transcribe(audio_path)
    text = result.get("text", "")

    # Gender detection
    gender = detect_gender(audio_path)

    os.remove(audio_path)
    return jsonify({"text": text, "gender": gender})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
