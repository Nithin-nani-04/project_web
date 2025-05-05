import streamlit as st
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, AutoTokenizer, AutoModelForSeq2SeqLM
import soundfile as sf
import os
import json

# Set Streamlit page configuration for a better user interface
st.set_page_config(page_title="Speech-to-Text and GEC", page_icon="🎙️", layout="wide")

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

# Registration function with user data saved in JSON file
def register_user(username, password):
    if os.path.exists("users.json"):
        with open("users.json", "r") as f:
            users = json.load(f)
    else:
        users = []
    users.append({"username": username, "password": password})

    with open("users.json", "w") as f:
        json.dump(users, f)

# Login function
def login_user(username, password):
    if os.path.exists("users.json"):
        with open("users.json", "r") as f:
            users = json.load(f)
        for user in users:
            if user["username"] == username and user["password"] == password:
                return True
    return False

# Streamlit UI for registration
def register_page():
    st.title("Create an Account")
    new_user = st.text_input("Enter Username", placeholder="Choose a unique username")
    new_pass = st.text_input("Enter Password", type="password", placeholder="Create a strong password")

    # Registration Button with an attractive design
    if st.button("Register", key="register_button"):
        if new_user and new_pass:
            register_user(new_user, new_pass)
            st.success(f"Account created successfully for {new_user}!", icon="✅")
        else:
            st.error("Both fields are required to register.", icon="❌")

# Streamlit UI for login
def login_page():
    st.title("Login to Your Account")
    username = st.text_input("Username", placeholder="Enter your username")
    password = st.text_input("Password", type="password", placeholder="Enter your password")

    # Login Button with visual improvements
    if st.button("Login", key="login_button"):
        if login_user(username, password):
            st.session_state.logged_in = True
            st.success(f"Welcome back {username}!", icon="✅")
        else:
            st.error("Invalid username or password. Please try again.", icon="❌")

# Audio processing page (ASR and GEC)
def audio_processing_page():
    st.title("Upload Audio for Transcription and Grammar Correction")
    
    # File uploader with interactive widget and tooltip
    uploaded_audio = st.file_uploader("Choose an audio file (wav, mp3, or flac)", type=["wav", "mp3", "flac"],
                                      help="Upload a clear audio file for transcription and grammar correction.")
    
    if uploaded_audio is not None:
        temp_path = f"/tmp/{uploaded_audio.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_audio.getbuffer())
        
        st.audio(temp_path, format="audio/wav")
        st.write("📝 Transcribing audio, please wait...")

        # Display loading spinner while transcription is happening
        with st.spinner("Processing..."):
            transcription = transcribe_audio(temp_path)

        st.write("**Transcription:**", transcription)

        # Grammar correction
        st.write("✏️ Correcting grammar...")
        with st.spinner("Correcting..."):
            corrected_transcription = correct_grammar(transcription)
        st.write("**Corrected Transcription:**", corrected_transcription)

# Main page selection
def main_page():
    # Sidebar with more visually appealing navigation
    st.sidebar.title("Navigation")
    menu = st.sidebar.radio("Select an Option", ["Login", "Register", "Upload Audio", "Logout"], index=0)

    if menu == "Register":
        register_page()
    elif menu == "Login":
        login_page()
    elif menu == "Upload Audio":
        if st.session_state.get("logged_in", False):
            audio_processing_page()
        else:
            st.warning("Please log in first to upload an audio file.")
    elif menu == "Logout":
        st.session_state.logged_in = False
        st.success("You have successfully logged out.", icon="✅")

# Footer with credits
def footer():
    st.markdown("""
        ---
        Created with ❤️ by AIML-10.
        """)

# Call the main page function to initialize the app
if __name__ == "__main__":
    # Ensure the session state is initialized correctly
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    main_page()
    footer()
