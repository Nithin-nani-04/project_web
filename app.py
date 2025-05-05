import streamlit as st
import whisper
import json
import os
import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load models once using caching
@st.cache_resource
def load_models():
    asr_model = whisper.load_model("base")  # Change to "small", "medium", or "large" if desired
    gec_tokenizer = AutoTokenizer.from_pretrained("gotutiyan/gec-t5-base-clang8")
    gec_model = AutoModelForSeq2SeqLM.from_pretrained("gotutiyan/gec-t5-base-clang8")
    return asr_model, gec_tokenizer, gec_model

asr_model, gec_tokenizer, gec_model = load_models()

# User credentials dictionary (for demo purposes)
users_db = {
    "user1": "pass1",
    "user2": "pass2"
}

# JSON logging
def save_to_json(username, audio_filename, transcription, corrected_text):
    record = {
        "timestamp": str(datetime.datetime.now()),
        "user": username,
        "audio_file": audio_filename,
        "transcription": transcription,
        "corrected_text": corrected_text
    }
    if os.path.exists("file.json"):
        with open("file.json", "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(record)
    with open("file.json", "w") as f:
        json.dump(data, f, indent=4)

# Whisper transcription
def transcribe_audio(file_path):
    result = asr_model.transcribe(file_path)
    return result["text"]

# Grammar correction
def correct_grammar(text):
    inputs = gec_tokenizer.encode("gec: " + text, return_tensors="pt", max_length=512, truncation=True)
    outputs = gec_model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    return gec_tokenizer.decode(outputs[0], skip_special_tokens=True)

# UI Logic
st.title("🔊 Speech-to-Text with Grammar Correction")

# User Login Section
st.sidebar.header("User Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
if st.sidebar.button("Login"):
    if username in users_db and users_db[username] == password:
        st.session_state["authenticated"] = True
        st.session_state["username"] = username
        st.success(f"Welcome, {username}!")
    else:
        st.error("Invalid username or password")

# Main App after Login
if st.session_state.get("authenticated", False):
    uploaded_file = st.file_uploader("📁 Upload Audio File (wav, mp3, m4a)", type=["wav", "mp3", "m4a"])
    if uploaded_file is not None:
        temp_path = "temp_audio." + uploaded_file.name.split('.')[-1]
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        st.audio(temp_path)
        st.write("📝 Transcribing...")
        transcription = transcribe_audio(temp_path)
        st.write("**Transcription:**", transcription)

        st.write("🔧 Correcting grammar...")
        corrected = correct_grammar(transcription)
        st.write("**Corrected Text:**", corrected)

        save_to_json(st.session_state["username"], uploaded_file.name, transcription, corrected)
        st.success("✅ Data saved to file.json")

# Registration info
with st.expander("Need an account?"):
    st.markdown("Ask admin to add your credentials directly to the `users_db` dictionary.")
