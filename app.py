from flask import Flask, render_template, request, jsonify
import tempfile
import librosa
import numpy as np
# from your_gender_model import predict_gender  # Replace with your gender model
# from whisper_module import transcribe_audio   # Replace with actual Whisper transcription

app = Flask(__name__)

# Dummy functions for demonstration
def predict_gender(y, sr):
    # Example: replace with actual model
    energy = np.mean(np.abs(y))
    return "Male" if energy > 0.01 else "Female"

def transcribe_audio(y, sr):
    # Example: replace with Whisper API call
    return "This is a dummy transcription."

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]

    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
        audio_file.save(temp_audio.name)
        y, sr = librosa.load(temp_audio.name, sr=16000)

        # Get transcription
        text = transcribe_audio(y, sr)

        # Get gender
        gender = predict_gender(y, sr)

        return jsonify({"text": text, "gender": gender})

if __name__ == "__main__":
    app.run(debug=True)
