import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from audio_recorder_streamlit import audio_recorder
import tempfile
import base64

# --- 1. CORE SETUP ---
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Groq LinguaFlow", page_icon="⚡", layout="centered")

# --- 2. API CONFIGURATION ---
if not api_key:
    st.error("❌ GROQ_API_KEY not found in .env file!")
    st.stop()

client = Groq(api_key=api_key)

# --- 3. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. UI DESIGN ---
st.title("⚡ Groq LinguaFlow")
st.markdown("Powered by **Whisper-large-v3-turbo**, **Llama-3.3-70b**, and **Orpheus TTS**")

with st.sidebar:
    st.header("Settings")
    target_lang = st.selectbox(
        "Translate AI Response to:",
        ["Urdu", "English", "Hindi", "Spanish", "French", "Arabic"],
        index=0
    )
    
    st.header("Voice Settings")
    # Using the latest supported personas for Orpheus-v1
    voice_persona = st.selectbox(
        "Voice Persona",
        ["hannah", "autumn", "diana", "austin", "daniel", "troy"],
        index=0,
        help="Select the voice character that fits your preference."
    )

    voice_emotion = st.selectbox(
        "Voice Emotion",
        ["neutral", "cheerful", "whisper", "excited", "sad"],
        index=0
    )

    if st.button("Clear Chat"):
        st.session_state.messages = []
        if "last_audio" in st.session_state:
            del st.session_state.last_audio
        st.rerun()

# --- 5. HELPER FUNCTIONS ---
def text_to_speech_groq(text, emotion="neutral", voice="hannah"):
    """Uses Groq's Orpheus TTS model"""
    try:
        # Formatting input with emotion tags if supported by the persona
        formatted_text = f"[{emotion}] {text}"
        
        response = client.audio.speech.create(
            model="canopylabs/orpheus-v1-english", 
            voice=voice,
            input=formatted_text,
            response_format="wav"
        )
        
        # Audio response comes as a stream/bytes
        audio_bytes = response.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
            fp.write(audio_bytes)
            return fp.name
            
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None

def autoplay_audio(file_path):
    """Auto-play audio in Streamlit using base64 encoding"""
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    md = f'<audio autoplay="true" src="data:audio/wav;base64,{b64}">'
    st.markdown(md, unsafe_allow_html=True)

def speech_to_text_groq(audio_bytes):
    """Uses Groq's Whisper-large-v3-turbo for fast transcription"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name

        with open(temp_audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(temp_audio_path, file.read()),
                model="whisper-large-v3-turbo",
                prompt="The audio contains English or Urdu phrases.",
                response_format="text",
            )
        os.unlink(temp_audio_path)
        return transcription
    except Exception as e:
        st.error(f"Speech Error: {e}")
        return None

# --- 6. DISPLAY HISTORY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- 7. INPUT METHODS ---
tab1, tab2 = st.tabs(["🎤 Voice Input", "⌨️ Text Input"])
user_query = None

with tab1:
    audio_data = audio_recorder(text="Click to record", icon_size="2x", key="recorder")
    if audio_data:
        if "last_audio" not in st.session_state or st.session_state.last_audio != audio_data:
            st.session_state.last_audio = audio_data
            with st.spinner("Transcribing..."):
                user_query = speech_to_text_groq(audio_data)

with tab2:
    with st.form("text_input_form", clear_on_submit=True):
        text_input = st.text_input("Type your message here...")
        submitted = st.form_submit_button("Send")
        if submitted and text_input:
            user_query = text_input

# --- 8. PROCESSING WITH LLAMA 3.3 ---
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)
    
    with st.spinner("Generating Response..."):
        try:
            chat_completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Instructions: Be extremely concise. "
                            f"1. Respond naturally in English. "
                            f"2. Provide a translation in {target_lang} at the end. "
                            f"3. Provide a one-sentence 'Summary' at the very bottom."
                        )
                    },
                    *st.session_state.messages
                ],
                max_tokens=300,
                temperature=0.6
            )
            
            ai_text = chat_completion.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": ai_text})
            
            with st.chat_message("assistant"):
                st.write(ai_text)
            
            # --- 9. VOICE OUTPUT ---
            # We only send the first sentence of the English response to TTS for speed
            speech_text = ai_text.split('\n')[0]
            
            with st.spinner("Synthesizing voice..."):
                audio_path = text_to_speech_groq(speech_text, voice_emotion, voice_persona)
                if audio_path:
                    autoplay_audio(audio_path)
                    os.unlink(audio_path)

        except Exception as e:
            st.error(f"Groq API Error: {e}")