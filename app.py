from flask import Flask, request, jsonify, render_template
import whisper
import torch
import torchaudio
import numpy as np
import librosa

app = Flask(__name__)

# Load Whisper model
model = whisper.load_model("base")

# Gender detection simple model (based on pitch)
def detect_gender(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches[pitches > 0])
    if pitch < 160:
        return "Male"
    else:
        return "Female"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio uploaded'}), 400
    
    audio_file = request.files['audio']
    file_path = 'temp.wav'
    audio_file.save(file_path)
    
    # Whisper transcription
    result = model.transcribe(file_path)
    text = result['text']
    
    # Gender detection
    gender = detect_gender(file_path)
    
    return jsonify({'text': text, 'gender': gender})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
