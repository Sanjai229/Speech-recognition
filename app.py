from flask import Flask, request, jsonify, render_template
import os
import speech_recognition as sr
# import your gender prediction model if you have one

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    temp_path = "temp_audio.wav"
    audio_file.save(temp_path)

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Placeholder for gender prediction
    gender = "Male/Female/Undefined"  # replace with your model logic

    return jsonify({'text': text, 'gender': gender})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
