from flask import Flask, request, jsonify
from pydub import AudioSegment
import whisper
import os

app = Flask(__name__)

# Load Whisper model
model = whisper.load_model("base")  # Change to "small", "medium", "large" if needed

@app.route('/')
def home():
    return "Whisper Flask API is running!"

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filename = file.filename
    file_path = os.path.join("/tmp", filename)
    file.save(file_path)

    # Convert audio to wav if needed
    audio = AudioSegment.from_file(file_path)
    wav_path = file_path.rsplit('.', 1)[0] + ".wav"
    audio.export(wav_path, format="wav")

    # Transcribe using Whisper
    result = model.transcribe(wav_path)
    text = result['text']

    # Clean up
    os.remove(file_path)
    os.remove(wav_path)

    return jsonify({"transcription": text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
