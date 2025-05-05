import streamlit as st
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, AutoTokenizer, AutoModelForSeq2SeqLM
import soundfile as sf
import os
import json
import re

# Load Wav2Vec2 model and processor for ASR (speech-to-text)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# Load T5 model and tokenizer for Grammar Error Correction (GEC)
tokenizer = AutoTokenizer.from_pretrained("gotutiyan/gec-t5-base-clang8")
gec_model = AutoModelForSeq2SeqLM.from_pretrained("gotutiyan/gec-t5-base-clang8")

# Utility function to transcribe audio (ASR)
def transcribe_audio(file_path):
    speech, _ = sf.read(file_path)
    input_values = processor(speech, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]

# Utility function to correct grammar (GEC) using T5
def correct_grammar(text):
    input_text = "grammar: " + text  # Adding task prefix for GEC
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = gec_model.generate(**inputs, max_length=512)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

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

# Feedback Validation
def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

def is_valid_phone(phone):
    return re.match(r"^\d{10}$", phone)  # Simple 10-digit validation

# Save feedback to JSON
def save_feedback_extended(name, email, phone, feedback):
    feedback_data = {
        "name": name,
        "email": email,
        "phone": phone,
        "feedback": feedback
    }
    with open("feedback.json", "a") as f:
        f.write(json.dumps(feedback_data) + "\n")

# Streamlit UI for feedback
def feedback_page():
    st.subheader("💬 Submit Feedback")

    name = st.text_input("Full Name")
    email = st.text_input("Email Address")
    phone = st.text_input("Phone Number")
    feedback = st.text_area("Your Feedback", height=150)

    if st.button("Submit Feedback"):
        errors = []

        if not name.strip():
            errors.append("Name is required.")
        if not is_valid_email(email):
            errors.append("Invalid email format.")
        if not is_valid_phone(phone):
            errors.append("Phone number must be exactly 10 digits.")
        if not feedback.strip():
            errors.append("Feedback cannot be empty.")

        if errors:
            for error in errors:
                st.error(error)
        else:
            save_feedback_extended(name, email, phone, feedback)
            st.success("✅ Thank you! Your feedback has been submitted.")

# Streamlit UI for registration
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

# Streamlit UI for login
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

# Audio processing page (ASR and GEC)
def audio_processing_page():
    st.title("Upload Audio for Transcription and GEC")
    uploaded_audio = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac"])
    
    if uploaded_audio is not None:
        temp_path = f"/tmp/{uploaded_audio.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_audio.getbuffer())
        
        st.audio(temp_path)
        st.write("📝 Transcribing...")
        
        # Step 1: Transcribe audio
        transcription = transcribe_audio(temp_path)
        st.write("**Transcription:**", transcription)
        
        # Step 2: Correct grammar of the transcription using T5
        st.write("✏️ Correcting grammar...")
        corrected_transcription = correct_grammar(transcription)
        st.write("**Corrected Transcription:**", corrected_transcription)

# Displaying Model Details
def display_model_details():
    st.markdown("### Model Details")

    st.subheader("Wav2Vec2 (ASR Model)")
    st.write(
        """
        - **Model Name**: Hybrid ASR Model
        - **Task**: Automatic Speech Recognition (ASR)
        - **Description**: Wav2Vec2 is a deep learning model that performs automatic speech recognition.
        - **Pretrained On**: 24 hours of English speech data.
        - **Input**: Raw audio waveform.
        - **Output**: Transcribed text.
        """
    )
    
    st.subheader("T5 (Grammar Error Correction Model)")
    st.write(
        """
        - **Model Name**: T5 (Text-to-Text Transfer Transformer)
        - **Task**: Grammar Error Correction (GEC)
        - **Description**: T5 is a transformer-based model capable of various text generation tasks, including grammar correction.
        - **Input**: Transcribed text.
        - **Output**: Corrected text with grammar improvements.
        """
    )

# Main page selection
def main_page():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    st.sidebar.title("Navigation")
    if st.session_state.logged_in:
        menu = st.sidebar.radio("Select an Option", ["Upload Audio", "Feedback", "Logout", "Model Details"])
    else:
        menu = st.sidebar.radio("Select an Option", ["Login", "Register"])

    if menu == "Register":
        register_page()
    elif menu == "Login":
        login_page()
    elif menu == "Upload Audio":
        audio_processing_page()
    elif menu == "Feedback":
        feedback_page()
    elif menu == "Logout":
        st.session_state.logged_in = False
        st.success("You have logged out successfully.")
    elif menu == "Model Details":
        display_model_details()

# Set Streamlit page configuration
st.set_page_config(page_title="Speech-to-Text and GEC", page_icon="📝", layout="wide")

if __name__ == "__main__":
    main_page()
