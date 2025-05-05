import streamlit as st
import torch
import json
import soundfile as sf
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import uuid

# Load ASR Model
asr_model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-large-librispeech-asr")
asr_processor = Speech2TextProcessor.from_pretrained("facebook/s2t-large-librispeech-asr")

# Load GEC Model
gec_model = AutoModelForSeq2SeqLM.from_pretrained("gotutiyan/gec-t5-base-clang8")
gec_tokenizer = AutoTokenizer.from_pretrained("gotutiyan/gec-t5-base-clang8")

# User credentials and data
USERS_FILE = "users.json"
DATA_FILE = "file.json"

def load_json(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return {}

def save_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

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

def transcribe_audio(audio_path):
    speech, _ = sf.read(audio_path)
    input_features = asr_processor(speech, sampling_rate=16000, return_tensors="pt").input_features
    generated_ids = asr_model.generate(input_features)
    transcription = asr_processor.batch_decode(generated_ids)[0]
    return transcription

def correct_text(text):
    inputs = gec_tokenizer.encode(text, return_tensors="pt")
    outputs = gec_model.generate(inputs, max_length=128, num_beams=5, early_stopping=True)
    corrected = gec_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected

# Streamlit UI
st.title("ASR + GEC Web App")

# Login/Register
auth_mode = st.sidebar.selectbox("Login / Register", ["Login", "Register"])
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if auth_mode == "Register":
    if st.sidebar.button("Register"):
        if register_user(username, password):
            st.sidebar.success("Registered successfully. Please log in.")
        else:
            st.sidebar.error("Username already exists.")

elif auth_mode == "Login":
    if st.sidebar.button("Login"):
        if authenticate_user(username, password):
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
        else:
            st.sidebar.error("Invalid credentials.")

if st.session_state.get("logged_in"):
    st.success(f"Welcome, {st.session_state['username']}!")

    uploaded_file = st.file_uploader("Upload an audio file (WAV)", type=["wav"])
    if uploaded_file:
        file_id = str(uuid.uuid4())
        file_path = f"audio/{file_id}.wav"
        os.makedirs("audio", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        transcription = transcribe_audio(file_path)
        st.subheader("ASR Transcription")
        st.write(transcription)

        corrected = correct_text(transcription)
        st.subheader("Grammatical Error Correction")
        st.write(corrected)

        # Save input/output data
        data_log = load_json(DATA_FILE)
        if username not in data_log:
            data_log[username] = []
        data_log[username].append({
            "original_transcription": transcription,
            "corrected_text": corrected,
            "file": uploaded_file.name
        })
        save_json(DATA_FILE, data_log)
