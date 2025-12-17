import whisper

model = whisper.load_model("tiny")

# USE A REAL WAV FILE (your voice, 5–10 sec)
audio_path = "test.wav"

result = model.transcribe(audio_path)
print("TRANSCRIPTION:")
print(result["text"])
