from flask import Flask, request, jsonify, render_template
import whisper
import numpy as np
import librosa
from pydub import AudioSegment
import os

app = Flask(__name__)

# Load Whisper model once
model = whisper.load_model("base")  # You can use small/medium/large

# Simple gender detection using pitch
def detect_gender(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    if len(pitch_values) == 0:
        return "Unknown"
    avg_pitch = np.mean(pitch_values)
    return "Female" if avg_pitch > 160 else "Male"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files['audio']
    audio_path = "temp_audio.wav"
    audio_file.save(audio_path)

    # Convert to mono WAV
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1)
    audio.export(audio_path, format="wav")

    # Whisper transcription
    result = model.transcribe(audio_path)
    text = result.get("text", "")

    # Gender detection
    gender = detect_gender(audio_path)

    # Remove temp audio
    os.remove(audio_path)

    return jsonify({
        "text": text,
        "gender": gender
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
