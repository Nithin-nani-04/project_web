import streamlit as st
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, AutoTokenizer, AutoModelForSeq2SeqLM
import soundfile as sf
import os
import json

st.set_page_config(page_title="🎙️ Voice GEC App", layout="wide")

# Load ASR model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# Load GEC model
tokenizer = AutoTokenizer.from_pretrained("gotutiyan/gec-t5-base-clang8")
gec_model = AutoModelForSeq2SeqLM.from_pretrained("gotutiyan/gec-t5-base-clang8")

# --------------------------- Utility Functions --------------------------- #

def transcribe_audio(file_path):
    speech, _ = sf.read(file_path)
    input_values = processor(speech, return_tensors="pt").input_values
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

# --------------------------- Pages --------------------------- #

def home_page():
    st.markdown("## 👋 Welcome to **Voice GEC** — Speech-to-Text with Grammar Correction")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎤 Audio Upload")
        st.markdown("Upload audio, and get grammatically corrected text output instantly!")

    with col2:
        st.markdown("### 🧠 Powered by AI")
        st.markdown("- ASR: Wav2Vec2\n- GEC: T5 model")
    
    st.info("Use the sidebar to **register**, **log in**, and start transcribing audio!")

def register_page():
    st.markdown("### 📝 Register")
    new_user = st.text_input("👤 Username")
    new_pass = st.text_input("🔒 Password", type="password")
    if st.button("Register"):
        if new_user and new_pass:
            register_user(new_user, new_pass)
            st.success(f"User **{new_user}** registered successfully!")
        else:
            st.error("Please enter both username and password.")

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
            st.error("Invalid credentials.")

def audio_processing_page():
    st.markdown("### 🎧 Upload Audio File")
    uploaded_audio = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac"])
    
    if uploaded_audio:
        temp_path = f"/tmp/{uploaded_audio.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_audio.read())

        st.audio(temp_path)
        with st.spinner("📝 Transcribing..."):
            transcription = transcribe_audio(temp_path)
        st.success("✅ Transcription complete!")
        st.markdown(f"**Transcript:** `{transcription}`")

        with st.spinner("✏️ Correcting grammar..."):
            corrected = correct_grammar(transcription)
        st.success("✅ Grammar correction complete!")
        st.markdown(f"**Corrected Text:** `{corrected}`")

def logout_page():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.success("You have been logged out.")

# --------------------------- Main App --------------------------- #

def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = ""

    st.sidebar.image("https://img.icons8.com/ios-filled/100/microphone.png", width=80)
    st.sidebar.title("📌 Navigation")

    if st.session_state.logged_in:
        menu = st.sidebar.radio("Choose an option:", ["🏠 Home", "🎤 Upload Audio", "🚪 Logout"])
        if menu == "🏠 Home":
            home_page()
        elif menu == "🎤 Upload Audio":
            audio_processing_page()
        elif menu == "🚪 Logout":
            logout_page()
    else:
        menu = st.sidebar.radio("Choose an option:", ["🏠 Home", "🔐 Login", "📝 Register"])
        if menu == "🏠 Home":
            home_page()
        elif menu == "🔐 Login":
            login_page()
        elif menu == "📝 Register":
            register_page()

main()
