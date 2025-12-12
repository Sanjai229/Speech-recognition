from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio uploaded'}), 400

    file = request.files['audio']
    audio_data = file.read()
    wav_io = io.BytesIO(audio_data)

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_io) as source:
        audio_file = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_file)
        except sr.UnknownValueError:
            text = "Could not understand audio"
        except sr.RequestError:
            text = "Could not request results"

    # Simple gender detection placeholder
    gender = "Male" if len(text) % 2 == 0 else "Female"

    return jsonify({'text': text, 'gender': gender})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
