import streamlit as st
import json
import os
import datetime
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ----------------- Config ------------------
st.set_page_config(page_title="ASR + GEC App", layout="centered")  # Set page config as the very first command

USERS_FILE = "users.json"
DATA_FILE = "file.json"

# ----------------- Model Load ------------------
@st.cache_resource
def load_models():
    asr = whisper.load_model("base")
    tokenizer = AutoTokenizer.from_pretrained("gotutiyan/gec-t5-base-clang8")
    gec_model = AutoModelForSeq2SeqLM.from_pretrained("gotutiyan/gec-t5-base-clang8")
    return asr, tokenizer, gec_model

asr_model, gec_tokenizer, gec_model = load_models()

# ----------------- User Management ------------------
def load_users():
    # Check if users file exists
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    else:
        # If the file doesn't exist, return an empty list
        return []

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def register_user(username, password):
    users = load_users()
    if any(user["username"] == username for user in users):
        return False
    users.append({"username": username, "password": password})
    save_users(users)
    return True

def authenticate_user(username, password):
    users = load_users()
    return any(user["username"] == username and user["password"] == password for user in users)

# ----------------- Whisper + GEC ------------------
def transcribe_audio(file_path):
    result = asr_model.transcribe(file_path)
    return result["text"]

def correct_grammar(text):
    inputs = gec_tokenizer.encode("gec: " + text, return_tensors="pt", max_length=512, truncation=True)
    outputs = gec_model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    return gec_tokenizer.decode(outputs[0], skip_special_tokens=True)

def save_data(username, filename, transcription, corrected_text):
    record = {
        "timestamp": str(datetime.datetime.now()),
        "user": username,
        "audio_file": filename,
        "transcription": transcription,
        "corrected_text": corrected_text
    }
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(record)
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ----------------- Pages ------------------

# Login Page
def login_page():
    st.subheader("🔓 Login")
    user = st.text_input("Username")
    passwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate_user(user, passwd):
            st.session_state.logged_in = True
            st.session_state.username = user
            st.success(f"Welcome, {user}!")
        else:
            st.error("Invalid credentials.")

# Registration Page
def register_page():
    st.subheader("🔐 Register")
    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type="password")
    if st.button("Create Account"):
        if register_user(new_user, new_pass):
            st.success("Account created! Please login.")
        else:
            st.error("Username already exists.")

# Audio Processing Page
def audio_processing_page():
    st.subheader(f"👋 Hello, {st.session_state.username}")
    uploaded = st.file_uploader("🎙️ Upload audio (wav, mp3, m4a)", type=["wav", "mp3", "m4a"])
    if uploaded:
        ext = uploaded.name.split('.')[-1]
        temp_path = f"temp_audio.{ext}"
        with open(temp_path, "wb") as f:
            f.write(uploaded.read())

        st.audio(temp_path)

        st.write("📝 Transcribing...")
        transcription = transcribe_audio(temp_path)
        st.write("**Transcription:**", transcription)

        st.write("✏️ Correcting grammar...")
        corrected = correct_grammar(transcription)
        st.write("**Corrected Text:**", corrected)

        save_data(st.session_state.username, uploaded.name, transcription, corrected)
        st.success("✅ Saved to file.json")

# Logout Page
def logout_page():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.success("Logged out!")
    st.experimental_rerun()

# ----------------- Main App ------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

# Sidebar Navigation
menu = st.sidebar.radio("Menu", ["Login", "Register", "Logout" if st.session_state.logged_in else None, "Upload Audio"] if st.session_state.logged_in else ["Login", "Register"])

# Display pages based on menu selection
if menu == "Login":
    login_page()
elif menu == "Register":
    register_page()
elif menu == "Upload Audio":
    if st.session_state.logged_in:
        audio_processing_page()
    else:
        st.warning("Please log in first.")
elif menu == "Logout":
    logout_page()
