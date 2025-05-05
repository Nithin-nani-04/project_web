import streamlit as st
import torch
import json
import os
import uuid
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- File paths ---
USERS_FILE = "users.json"
DATA_FILE = "file.json"
os.makedirs("audio", exist_ok=True)

# --- Load Models ---
@st.cache_resource
def load_models():
    asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    asr_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
    gec_model = AutoModelForSeq2SeqLM.from_pretrained("gotutiyan/gec-t5-base-clang8")
    gec_tokenizer = AutoTokenizer.from_pretrained("gotutiyan/gec-t5-base-clang8")
    return asr_model, asr_tokenizer, gec_model, gec_tokenizer

asr_model, asr_tokenizer, gec_model, gec_tokenizer = load_models()

# --- Utils ---
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

def authenticate(username, password):
    users = load_json(USERS_FILE)
    return users.get(username) == password

def transcribe_audio(path):
    speech, _ = sf.read(path)
    input_values = asr_tokenizer(speech, return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = asr_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return asr_tokenizer.batch_decode(predicted_ids)[0]

def correct_text(text):
    inputs = gec_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = gec_model.generate(**inputs, max_length=128)
    return gec_tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Streamlit UI ---
st.title("ASR + Grammar Correction App (Local Models)")

auth_choice = st.sidebar.radio("Choose", ["Login", "Register"])
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if auth_choice == "Register":
    if st.sidebar.button("Register"):
        if register_user(username, password):
            st.success("User registered!")
        else:
            st.error("Username already exists.")

elif auth_choice == "Login":
    if st.sidebar.button("Login"):
        if authenticate(username, password):
            st.session_state["user"] = username
            st.success(f"Welcome {username}!")
        else:
            st.error("Invalid credentials.")

# --- Main App After Login ---
if "user" in st.session_state:
    st.subheader("Upload a WAV audio file for transcription and correction:")
    uploaded = st.file_uploader("Audio File", type=["wav"])
    if uploaded:
        file_id = str(uuid.uuid4())
        file_path = f"audio/{file_id}.wav"
        with open(file_path, "wb") as f:
            f.write(uploaded.read())

        st.info("Transcribing...")
        transcription = transcribe_audio(file_path)
        st.write("**Transcription:**", transcription)

        st.info("Correcting grammar...")
        corrected = correct_text(transcription)
        st.write("**Corrected Text:**", corrected)

        # Save to data log
        log = load_json(DATA_FILE)
        log.setdefault(st.session_state["user"], []).append({
            "file": uploaded.name,
            "transcription": transcription,
            "corrected": corrected
        })
        save_json(DATA_FILE, log)
        st.success("Data saved.")
