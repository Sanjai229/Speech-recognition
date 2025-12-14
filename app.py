from flask import Flask, render_template, request, jsonify
import whisper
from io import BytesIO
from pydub import AudioSegment
import tempfile
import os

app = Flask(__name__)
model = whisper.load_model("base")  # You can switch to "small", "medium", "large" if needed

# Simple gender detection based on pronouns (demo purpose)
def predict_gender(text):
    male_words = ['he', 'him', 'his']
    female_words = ['she', 'her', 'hers']
    text_lower = text.lower()
    if any(word in text_lower for word in male_words):
        return "Male"
    elif any(word in text_lower for word in female_words):
        return "Female"
    else:
        return "Unknown"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'text': 'No audio file', 'gender': 'Undefined'})

    audio_file = request.files['audio']
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            audio_segment = AudioSegment.from_file(audio_file)
            audio_segment.export(tmp.name, format="wav")
            tmp_path = tmp.name

        # Transcribe using Whisper
        result = model.transcribe(tmp_path)
        text = result["text"].strip()

        # Predict gender
        gender = predict_gender(text)

        # Cleanup temp file
        os.remove(tmp_path)

        return jsonify({'text': text, 'gender': gender})
    except Exception as e:
        print("Error:", e)
        return jsonify({'text': 'Error', 'gender': 'Undefined'})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
