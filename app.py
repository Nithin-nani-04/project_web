import streamlit as st
import torch
import json
import os
import uuid
import soundfile as sf
from transformers import (
    Wav2Vec2Processor, Wav2Vec2ForCTC,
    AutoTokenizer, AutoModelForSeq2SeqLM
)

# -------------------- CONFIG --------------------
USERS_FILE = "users.json"
DATA_FILE = "file.json"
AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_models():
    asr_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    gec_tokenizer = AutoTokenizer.from_pretrained("gotutiyan/gec-t5-base-clang8")
    gec_model = AutoModelForSeq2SeqLM.from_pretrained("gotutiyan/gec-t5-base-clang8")
    return asr_processor, asr_model, gec_tokenizer, gec_model

asr_processor, asr_model, gec_tokenizer, gec_model = load_models()

# -------------------- UTILITIES --------------------
def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else {}

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def register_user(username, password):
    users = load_json(USERS_FILE)
    if username in users:
        return False
    users[username] = password
    save_json(USERS_FILE, users)
    return True

def authenticate_user(username, password):
    users = load_json(USERS_FILE)
    return users.get(username) == password

def transcribe_audio(file_path):
    speech, _ = sf.read(file_path)
    inputs = asr_processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = asr_model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def correct_text(text):
    inputs = gec_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = gec_model.generate(**inputs, max_length=128)
    return gec_tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------- STREAMLIT APP --------------------
st.title("🎤 Local ASR + Grammar Correction Web App")

# Sidebar: Login/Register
option = st.sidebar.radio("Choose", ["Login", "Register"])
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if option == "Register":
    if st.sidebar.button("Register"):
        if register_user(username, password):
            st.success("User registered! You can now log in.")
        else:
            st.error("Username already exists.")

if option == "Login":
    if st.sidebar.button("Login"):
        if authenticate_user(username, password):
            st.session_state["user"] = username
            st.success(f"Welcome, {username}!")
        else:
            st.error("Invalid username or password.")

# Main content if logged in
if "user" in st.session_state:
    st.subheader("Upload your audio file (WAV 16kHz mono)")
    uploaded_file = st.file_uploader("Choose a file", type=["wav"])

    if uploaded_file is not None:
        file_id = str(uuid.uuid4())
        save_path = os.path.join(AUDIO_DIR, f"{file_id}.wav")
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())

        st.audio(save_path, format='audio/wav')

        st.info("Transcribing...")
        transcription = transcribe_audio(save_path)
        st.write("**Transcription:**")
        st.success(transcription)

        st.info("Correcting grammar...")
        corrected = correct_text(transcription)
        st.write("**Corrected Text:**")
        st.success(corrected)

        # Save data per user
        log = load_json(DATA_FILE)
        log.setdefault(st.session_state["user"], []).append({
            "file": uploaded_file.name,
            "transcription": transcription,
            "corrected": corrected
        })
        save_json(DATA_FILE, log)

        st.success("Saved successfully!")

