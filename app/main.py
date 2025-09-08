import os
import math
import tempfile
import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma

from transformers import MarianMTModel, MarianTokenizer
import azure.cognitiveservices.speech as speechsdk

from pydub import AudioSegment

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SERVICE_REGION = os.getenv("AZURE_SERVICE_REGION")

# Path to background music
BGM_PATH = os.getenv("BGM_PATH", "assets/bgm.mp3")

# Load translation model
translation_model_name = "Helsinki-NLP/opus-mt-en-lg"
translator_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translator_model = MarianMTModel.from_pretrained(translation_model_name)

def translate_to_luganda(text: str) -> str:
    try:
        tokens = translator_tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = translator_model.generate(**tokens)
        return translator_tokenizer.decode(translated[0], skip_special_tokens=True)
    except Exception as e:
        return f"[Translation error: {e}]"

# Audio helpers (TTS + Music)
def _norm(audio: AudioSegment) -> AudioSegment:
    """Ensure consistent sample rate & channels (stereo, 44.1k)."""
    return audio.set_frame_rate(44100).set_channels(2)

def _loop_to_length(track: AudioSegment, target_ms: int) -> AudioSegment:
    """Loop/trim a track to exactly match target length."""
    if len(track) == 0:
        return track
    if len(track) >= target_ms:
        return track[:target_ms]
    loops = math.ceil(target_ms / len(track))
    out = sum([track] * loops)
    return out[:target_ms]

def mix_tts_with_music(tts_path: str, music_path: str, out_path: str, music_gain_db: float = -18.0) -> str:
    """
    Mix the TTS narration with background music.
    - music_gain_db: negative values reduce music volume (e.g., -18 dB is subtle)
    """
    try:
        voice = _norm(AudioSegment.from_file(tts_path))
        music = _norm(AudioSegment.from_file(music_path))
        music = music + music_gain_db
        music = _loop_to_length(music, len(voice))

        mixed = music.overlay(voice)
        mixed.export(out_path, format="mp3")
        return out_path
    except Exception as e:
        shutil.copy(tts_path, out_path)
        return out_path

def azure_tts_to_file(text: str, language: str, out_path: str) -> str:
    """Create TTS narration with Azure into out_path."""
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SERVICE_REGION)
    if language == "Luganda":
        speech_config.speech_synthesis_voice_name = "lg-UG-ApolloNeural"
    else:
        speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"

    audio_config = speechsdk.audio.AudioOutputConfig(filename=out_path)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    synthesizer.speak_text_async(text).get()
    return out_path

def safe_generate_narration(text: str, language: str, want_music: bool, music_gain_db: float) -> str:
    """
    Generate an mp3 narration. If want_music and BGM exists, mix it under the voice.
    Returns the final file path.
    """
    tts_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    azure_tts_to_file(text, language, tts_path)

    final_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name

    if want_music and os.path.exists(BGM_PATH):
        return mix_tts_with_music(tts_path, BGM_PATH, final_path, music_gain_db=music_gain_db)
    shutil.copy(tts_path, final_path)
    return final_path

# Embeddings + Vectorstore
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(
    persist_directory="story_db",
    embedding_function=embeddings
)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# Groq LLM + Memory
llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=GROQ_API_KEY)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)

# Streamlit Setup
st.set_page_config(page_title="📖 Story Chatbot", layout="wide")
st.title("📄Story-Bot💬")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None

# Tabs for Modes
tab_story, tab_chat = st.tabs(["📚 Story Library", "💬 Chat Mode"])

# Global Controls
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 2])
with col1:
    language = st.radio("🌍 Language", ["English", "Luganda"], horizontal=True)
with col2:
    bgm_toggle = st.checkbox("🎵 Background music", value=True, help="Mix soft music under the narration")
with col3:
    music_level = st.slider("🎚️ Music volume (quieter ⟶ louder)", min_value=-30, max_value=-6, value=-18, step=1)

# Story Library Tab
with tab_story:
    st.header("📖 Story Library")

    # Fetch all docs from vectordb
    all_docs = vectordb._collection.get(include=["metadatas", "documents"])
    story_titles = [meta["title"] for meta in all_docs["metadatas"] if "title" in meta]

    for idx, title in enumerate(story_titles):
        preview_words = all_docs["documents"][idx].split()[:50]
        preview_text = " ".join(preview_words) + "..."

        with st.expander(f"📖 {title}"):
            st.markdown(preview_text)
            if st.button(f"Read Full Story: {title}", key=f"read_{idx}"):
                full_text = all_docs["documents"][idx]
                final_text = translate_to_luganda(full_text) if language == "Luganda" else full_text

                # Narration
                voice_file = safe_generate_narration(final_text, language, want_music=bgm_toggle, music_gain_db=music_level)

                bot_msg = AIMessage(content=final_text)
                bot_msg.audio_path = voice_file
                st.session_state.messages.append(bot_msg)

# Chat Mode Tab
with tab_chat:
    user_input = st.chat_input("Ask for a story or a question...")
    if user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))

        try:
            response = qa({"question": user_input})
            answer = response.get("answer", "🤖 No answer returned.")
            final_answer = translate_to_luganda(answer) if language == "Luganda" else answer

            voice_file = safe_generate_narration(final_answer, language, want_music=bgm_toggle, music_gain_db=music_level)

            bot_msg = AIMessage(content=final_answer)
            bot_msg.audio_path = voice_file
            st.session_state.messages.append(bot_msg)

        except Exception as e:
            st.session_state.messages.append(AIMessage(content=f"❌ Error: {e}"))

# Display Chat Messages
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)
        if hasattr(msg, "audio_path"):
            st.audio(msg.audio_path, format="audio/mp3")

# Clear Chat Button
if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.session_state.quiz_data = None
    memory.clear()
