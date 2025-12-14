from flask import Flask, render_template, request, jsonify
import whisper
from io import BytesIO
from pydub import AudioSegment

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
        audio_bytes = BytesIO(audio_file.read())
        audio_segment = AudioSegment.from_file(audio_bytes)
        wav_bytes = BytesIO()
        audio_segment.export(wav_bytes, format="wav")
        wav_bytes.seek(0)

        # Transcribe
        result = model.transcribe(wav_bytes)
        text = result["text"].strip()

        # Predict gender
        gender = predict_gender(text)

        return jsonify({'text': text, 'gender': gender})
    except Exception as e:
        print("Error:", e)
        return jsonify({'text': 'Error', 'gender': 'Undefined'})

if __name__ == "__main__":
    app.run(debug=True)
