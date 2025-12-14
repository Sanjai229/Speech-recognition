from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import tempfile
import os
import subprocess

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400

    audio_file = request.files["audio"]

    # Save original browser audio (webm/ogg)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as input_audio:
        audio_file.save(input_audio.name)
        input_path = input_audio.name

    # Convert to WAV using ffmpeg
    output_path = input_path.replace(".webm", ".wav")

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, output_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        recognizer = sr.Recognizer()
        with sr.AudioFile(output_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)

    except sr.UnknownValueError:
        text = "Could not understand audio"
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)

    return jsonify({"text": text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
