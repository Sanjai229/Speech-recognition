from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import os
import tempfile

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file received"}), 400

    audio_file = request.files["audio"]

    recognizer = sr.Recognizer()

    # Save audio temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        audio_file.save(temp_audio.name)
        temp_path = temp_audio.name

    try:
        with sr.AudioFile(temp_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        text = "Could not understand the audio"
    except sr.RequestError as e:
        return jsonify({"error": f"Speech service error: {e}"}), 500
    finally:
        os.remove(temp_path)

    return jsonify({"text": text})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
