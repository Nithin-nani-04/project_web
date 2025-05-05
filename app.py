import os
import json
import shutil
import streamlit as st
import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from keras.models import load_model
import torch

# ----------------- Authentication Functions -----------------
def load_users():
    if os.path.exists('user.json'):
        with open('user.json', 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open('user.json', 'w') as f:
        json.dump(users, f, indent=4)

def authenticate(username, password):
    users = load_users()
    return users.get(username) == password

def register_user(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = password
    save_users(users)
    return True

# ----------------- Logging Functions -----------------
def log_file_info(user, filename, original_text, corrected_text):
    data = {
        "user": user,
        "file": filename,
        "original_text": original_text,
        "corrected_text": corrected_text
    }
    if os.path.exists('file.json'):
        with open('file.json', 'r') as f:
            try:
                file_log = json.load(f)
            except:
                file_log = []
    else:
        file_log = []

    file_log.append(data)
    with open('file.json', 'w') as f:
        json.dump(file_log, f, indent=4)

def log_feedback(user, feedback):
    data = {"user": user, "feedback": feedback}
    with open('feedback.json', 'a') as f:
        json.dump(data, f)
        f.write("\n")

# ----------------- Model Loading -----------------
asr_model = load_model('ASR2_Model.h5', compile=False)
tokenizer = AutoTokenizer.from_pretrained("gotutiyan/gec-t5-base-clang8")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("gotutiyan/gec-t5-base-clang8")
vocabulary = [""] + [chr(i) for i in range(97, 97 + 26)] + [".", ",", "?", " "]
FRAME_LENGTH = 255
FRAME_STEP = 128

# ----------------- ASR + GEC Functions -----------------
def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

def get_spec_inference(filepath):
    audio_binary = tf.io.read_file(filepath)
    waveform = decode_audio(audio_binary)
    waveform = tf.cast(waveform, tf.float32)
    spectrogram = tf.signal.stft(waveform, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP)
    spectrogram = tf.abs(spectrogram)
    return tf.expand_dims(spectrogram, axis=-1)

def decode(y_pred):
    batch_size = tf.shape(y_pred)[0]
    pred_length = tf.shape(y_pred)[1]
    pred_length *= tf.ones([batch_size], dtype=tf.int32)
    y_pred = tf.one_hot(y_pred, len(vocabulary) + 1)
    output = tf.keras.backend.ctc_decode(y_pred, input_length=pred_length, greedy=True)[0][0]
    out = [vocabulary[i] for i in output[0]]
    return ''.join(out)

def correct_grammar_with_t5(text):
    encoding = tokenizer(text, padding="max_length", max_length=256, truncation=True, return_tensors='pt')
    input_ids = encoding['input_ids'].to(torch.device('cpu'))
    outputs = t5_model.generate(input_ids, max_length=256, num_beams=8, do_sample=True, eos_token_id=tokenizer.eos_token_id)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="ASR + GEC App", layout="centered")

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""

# ----------------- Sidebar: Logout + Navigation -----------------
st.sidebar.title("User Menu")

if st.session_state.get("authenticated", False):
    st.sidebar.write(f"👤 Logged in as: `{st.session_state.username}`")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.success("You have been logged out.")
        # Optional: Clean uploads
        if os.path.exists("uploads"):
            shutil.rmtree("uploads")

if st.session_state.get("authenticated", False):
    menu = st.sidebar.selectbox("Navigation", ["App", "Feedback"])
else:
    menu = st.sidebar.selectbox("Navigation", ["Login", "Register"])

# ----------------- Login Page -----------------
if menu == "Login":
    st.header("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate(username, password):
            st.success("✅ Logged in successfully.")
            st.session_state.authenticated = True
            st.session_state.username = username
        else:
            st.error("❌ Invalid username or password.")

# ----------------- Register Page -----------------
elif menu == "Register":
    st.header("📝 Register")
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    if st.button("Register"):
        if register_user(username, password):
            st.success("✅ Registration successful! Please log in.")
        else:
            st.error("❌ Username already exists.")

# ----------------- App Page -----------------
elif menu == "App":
    if not st.session_state.get("authenticated", False):
        st.warning("⚠️ Please log in first.")
    else:
        st.title('🎤 ASR + T5 GEC: Audio Transcription and Grammar Correction')
        uploaded_file = st.file_uploader("Upload an Audio File (WAV, MP3)", type=["wav", "mp3"])

        if uploaded_file is not None:
            file_path = os.path.join('uploads', uploaded_file.name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.info("⏳ Processing your audio file...")

            spectrogram = get_spec_inference(file_path)
            prediction = asr_model.predict(tf.expand_dims(spectrogram, axis=0))
            out = tf.argmax(prediction[0], axis=1)
            decoded_text = decode(tf.expand_dims(out, axis=0))

            st.subheader("🗣️ Transcription:")
            st.write(decoded_text)

            corrected_text = correct_grammar_with_t5(decoded_text)
            st.subheader("📝 Corrected Text:")
            st.write(corrected_text)

            log_file_info(st.session_state.username, uploaded_file.name, decoded_text, corrected_text)

# ----------------- Feedback Page -----------------
elif menu == "Feedback":
    if not st.session_state.get("authenticated", False):
        st.warning("⚠️ Please log in first.")
    else:
        st.header("💬 Feedback")
        feedback_text = st.text_area("Write your feedback below:")
        if st.button("Submit Feedback"):
            if feedback_text.strip():
                log_feedback(st.session_state.username, feedback_text.strip())
                st.success("🙏 Thank you for your feedback!")
            else:
                st.error("❌ Feedback cannot be empty.")
