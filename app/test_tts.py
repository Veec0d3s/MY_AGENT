from TTS.api import TTS

# Option 1: Use an instance to list models
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
print("Model loaded:", tts.model_name)

# Convert text to audio file
tts.tts_to_file(text="Hello world! This is a test.", file_path="test_output.wav")
print("Audio saved to test_output.wav")

# Option 2: List all available models (without creating an instance)
from TTS.utils.manage import list_models
print("Available models:", list_models())
