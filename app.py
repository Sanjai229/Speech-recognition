from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import numpy as np

app = Flask(__name__)

def predict_gender(audio_data):
    """
    Simple gender prediction using pitch estimation
    (This is a placeholder; you can replace with ML model if needed)
    """
    # Convert raw audio to numpy array
    y = np.frombuffer(audio_data, np.int16)
    # Estimate pitch using FFT
    fft = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(fft))
    magnitude = np.abs(fft)
    peak_freq = abs(freqs[np.argmax(magnitude)])
    
    # Simple threshold to predict male/female
    if peak_freq < 0.1:
        return "Male"
    else:
        return "Female"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    file = request.files["audio"]
    recognizer = sr.Recognizer()

    with sr.AudioFile(file) as source:
        audio = recognizer.record(source)

    # Convert to text using Google API
    try:
        text = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        text = "Could not understand audio"
    except sr.RequestError:
        text = "Could not request results from Google API"

    # Gender prediction
    audio_data = audio.get_raw_data()
    gender = predict_gender(audio_data)

    return jsonify({"text": text, "gender": gender})

if __name__ == "__main__":
    app.run(debug=True)
