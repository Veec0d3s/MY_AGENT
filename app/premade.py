import streamlit as st
from TTS.api import TTS

# Load TTS model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

st.title("Hera Story Narrator")

# Example categories
categories = {
    "Resilience": [
        "A young girl overcomes obstacles in her village and becomes a community leader.",
        "An inventor fails many times but finally creates a life-changing device."
    ],
    "Friendship": [
        "Two friends solve a mystery together in their town.",
        "A lost dog brings neighbors closer and teaches them kindness."
    ],
    "Adventure": [
        "A journey across mountains to find a hidden treasure filled with secrets.",
        "Exploring an abandoned city full of mysteries and surprises."
    ]
}

# Select category
category = st.selectbox("Choose a story category:", list(categories.keys()))

# Select story from category
story = None
if category:
    story = st.selectbox("Pick a story:", categories[category])

# Display story text always
if story:
    st.subheader("Story Text (readable version):")
    st.write(story)

    # Optional: Button to listen
    if st.button("Narrate Story"):
        audio_file = "story_audio.wav"
        tts.tts_to_file(text=story, file_path=audio_file)
        st.audio(audio_file, format="audio/wav")
