import os
os.environ["PYTORCH_NO_LAZY_INIT"] = "1"  # Prevent meta-device loading

import streamlit as st
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, AutoTokenizer, AutoModelForSeq2SeqLM
import soundfile as sf
import json

st.set_page_config(page_title="🎙️ Voice GEC App", layout="wide")

# Load models (Wav2Vec2 and GEC)
ASR_MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
GEC_MODEL_NAME = "gotutiyan/gec-t5-base-clang8"

@st.cache_resource
def load_models():
    processor = Wav2Vec2Processor.from_pretrained(ASR_MODEL_NAME)
    asr_model = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(GEC_MODEL_NAME)
    gec_model = AutoModelForSeq2SeqLM.from_pretrained(GEC_MODEL_NAME)
    return processor, asr_model, tokenizer, gec_model

processor, asr_model, tokenizer, gec_model = load_models()

# ---------------- Utility Functions ---------------- #

def transcribe_audio(file_path):
    speech, _ = sf.read(file_path)
    input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        logits = asr_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]

def correct_grammar(text):
    input_text = "grammar: " + text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = gec_model.generate(**inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def register_user(username, password):
    if os.path.exists("users.json"):
        with open("users.json", "r") as f:
            users = json.load(f)
    else:
        users = []
    users.append({"username": username, "password": password})
    with open("users.json", "w") as f:
        json.dump(users, f)

def login_user(username, password):
    if os.path.exists("users.json"):
        with open("users.json", "r") as f:
            users = json.load(f)
        for user in users:
            if user["username"] == username and user["password"] == password:
                return True
    return False

# ---------------- UI Pages ---------------- #

def home_page():
    st.title("🎙️ Voice Grammar Correction App")
    st.markdown("Convert your speech into grammatically correct text using AI.")
    st.info("Use the sidebar to log in, register, or upload your audio.")

def register_page():
    st.markdown("### 📝 Register")
    new_user = st.text_input("👤 Username")
    new_pass = st.text_input("🔒 Password", type="password")
    if st.button("Register"):
        if new_user and new_pass:
            register_user(new_user, new_pass)
            st.success(f"User **{new_user}** registered successfully!")
        else:
            st.error("Both fields are required.")

def login_page():
    st.markdown("### 🔐 Login")
    username = st.text_input("👤 Username")
    password = st.text_input("🔒 Password", type="password")
    if st.button("Login"):
        if login_user(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome back, **{username}**!")
        else:
            st.error("Invalid username or password.")

def audio_processing_page():
    st.markdown("### 🎧 Upload Your Audio File")
    uploaded_audio = st.file_uploader("Choose an audio file", type=["wav", "flac", "mp3"])
    
    if uploaded_audio:
        temp_path = f"/tmp/{uploaded_audio.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_audio.read())

        st.audio(temp_path, format="audio/wav")

        with st.spinner("🔍 Transcribing..."):
            transcription = transcribe_audio(temp_path)
        st.success("✅ Transcription Complete")
        st.markdown(f"**Transcript:** `{transcription}`")

        with st.spinner("✏️ Correcting grammar..."):
            corrected = correct_grammar(transcription)
        st.success("✅ Grammar Correction Complete")
        st.markdown(f"**Corrected Text:** `{corrected}`")

def logout_page():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.success("🚪 You have logged out.")

# ---------------- Main App ---------------- #

def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = ""

    st.sidebar.image("https://img.icons8.com/ios-filled/100/microphone.png", width=80)
    st.sidebar.title("📌 Navigation")

    if st.session_state.logged_in:
        menu = st.sidebar.radio("Menu", ["🏠 Home", "🎤 Upload Audio", "🚪 Logout"])
        if menu == "🏠 Home":
            home_page()
        elif menu == "🎤 Upload Audio":
            audio_processing_page()
        elif menu == "🚪 Logout":
            logout_page()
    else:
        menu = st.sidebar.radio("Menu", ["🏠 Home", "🔐 Login", "📝 Register"])
        if menu == "🏠 Home":
            home_page()
        elif menu == "🔐 Login":
            login_page()
        elif menu == "📝 Register":
            register_page()

main()
