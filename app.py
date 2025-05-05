import streamlit as st
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
import json
import os

# Load Wav2Vec2 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# Utility function to transcribe audio
def transcribe_audio(file_path):
    speech, _ = sf.read(file_path)
    input_values = processor(speech, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]

# Registration
def register_user(username, password):
    if os.path.exists("users.json"):
        with open("users.json", "r") as f:
            users = json.load(f)
    else:
        users = []

    users.append({"username": username, "password": password})

    with open("users.json", "w") as f:
        json.dump(users, f)

# Login
def login_user(username, password):
    if os.path.exists("users.json"):
        with open("users.json", "r") as f:
            users = json.load(f)
        for user in users:
            if user["username"] == username and user["password"] == password:
                return True
    return False

# Streamlit UI
def register_page():
    st.title("Register")
    new_user = st.text_input("New Username")
    new_pass = st.text_input("New Password", type="password")
    if st.button("Register"):
        if new_user and new_pass:
            register_user(new_user, new_pass)
            st.success(f"User {new_user} registered successfully!")
        else:
            st.error("Please enter both username and password.")

def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login_user(username, password):
            st.session_state.logged_in = True
            st.success(f"Welcome back {username}!")
        else:
            st.error("Invalid username or password.")

def audio_processing_page():
    st.title("Upload Audio for Transcription")
    uploaded_audio = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac"])
    
    if uploaded_audio is not None:
        temp_path = f"/tmp/{uploaded_audio.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_audio.getbuffer())
        
        st.audio(temp_path)
        st.write("📝 Transcribing...")
        transcription = transcribe_audio(temp_path)
        st.write("**Transcription:**", transcription)

# Main page selection
def main_page():
    st.sidebar.title("Navigation")
    menu = st.sidebar.radio("Select an Option", ["Login", "Register", "Upload Audio", "Logout"])

    if menu == "Register":
        register_page()
    elif menu == "Login":
        login_page()
    elif menu == "Upload Audio":
        if st.session_state.get("logged_in", False):
            audio_processing_page()
        else:
            st.warning("Please log in first.")
    elif menu == "Logout":
        st.session_state.logged_in = False
        st.success("You have logged out successfully.")

# Set Streamlit page configuration
st.set_page_config(page_title="Speech-to-Text", page_icon="📝", layout="wide")

if __name__ == "__main__":
    main_page()
