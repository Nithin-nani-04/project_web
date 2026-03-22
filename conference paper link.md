https://ieeexplore.ieee.org/document/11071202
# 🎙️ Hybrid ASR Model with T5 for Audio-to-Text Enhancement

## 📄 Paper Details

* **Title:** A Hybrid ASR Model Integrating with T5 for Enhancement of Audio Text
* **Conference:** IEEE (2025) 
* **Authors:**

  * Mr. N. Bala Krishna
  * Daggolu Laharika
  * Ms. Vidhyavathi Kotha
  * **Pothireddy Nithin Kumar Reddy**
  * Beechu Sowri Nandan Reddy
  * P Jaswanth

---

## 🚀 Overview

This project presents a **two-stage hybrid system** that converts audio into high-quality, grammatically correct text.

👉 Traditional ASR systems generate raw transcripts but struggle with:

* Grammar errors
* Poor sentence structure
* Lack of contextual understanding

✅ This system solves that by combining:

1. **ASR Model (Speech → Raw Text)**
2. **T5 Transformer (Raw Text → Refined Text)**

---

## 🎯 Key Contributions

* 🔹 Reduced **Word Error Rate (WER) by ~7%**
* 🔹 Improved **BLEU score by ~4%**
* 🔹 Achieved **context-aware grammatical correction**
* 🔹 Designed a **real-time applicable pipeline**

---

## 🧠 System Architecture

### 🔹 Stage 1: ASR Module

* Inspired by **DeepSpeech2 architecture**
* Converts audio → spectrogram → text

#### Key Components:

* Convolutional Layers (feature extraction)
* Bidirectional GRU Layers (context understanding)
* LSTM Layer (sequence refinement)
* CTC Loss (alignment)

📊 Dataset:

* **LJ Speech Dataset (~13,100 samples)**

📉 Performance:

* WER = **10.91%** 

---

### 🔹 Stage 2: GEC Module (T5-Based)

* Uses **T5-Base Transformer**
* Converts raw text → grammatically correct text

#### Features:

* Encoder-Decoder Architecture
* Text-to-Text paradigm
* Pretrained on **C4 dataset**
* Fine-tuned on:

  * GEC Corpus
  * Lang-8 Dataset

📊 Training:

* 1M training samples
* 50K testing samples
* 5 epochs

📈 Performance:

* BLEU Score ≈ **72%**
* Significant grammar improvement 

---

## 🔄 Pipeline Workflow

```
Audio Input
   ↓
Spectrogram (STFT)
   ↓
ASR Model (GRU + LSTM + CTC)
   ↓
Raw Text
   ↓
T5 Model (Grammar Correction)
   ↓
Final Refined Text
```

---

## ⚙️ Technologies Used

* Python
* Deep Learning (RNN, GRU, LSTM)
* Transformers (T5)
* Librosa (Audio Processing)
* HuggingFace Transformers

---

## 📊 Evaluation Metrics

### 1. Word Error Rate (WER)

* Measures transcription accuracy
* ASR Output: **0.1091**

### 2. BLEU Score

* Measures grammatical correctness
* Post-processing improves fluency significantly

---

## 📈 Results Summary

| Model                   | WER       | BLEU      |
| ----------------------- | --------- | --------- |
| Google Cloud ASR        | 38.77     | 53.39     |
| IBM Watson              | 40.54     | 51.59     |
| Hybrid ASR              | 20.26     | 72.77     |
| **Proposed (ASR + T5)** | **10.91** | **78.76** |

📌 Shows strong improvement over baseline systems 

---

## 🌍 Applications

* 🎧 Speech-to-text systems
* 🏥 Medical transcription
* ⚖️ Legal documentation
* 🎓 Educational tools
* ♿ Accessibility systems

---

## ⚠️ Limitations

* Requires high computational resources
* Performance may drop in noisy environments
* Slight trade-off between BLEU and WER

---

## 🔮 Future Work

* Multi-language support
* Real-time deployment
* Integration with LLMs (e.g., Falcon, GPT)
* Larger transformer models (T5-Large)

---

## 🧩 Key Insight

👉 The real innovation is NOT ASR alone
👉 It’s the **combination of ASR + NLP (T5)**

This bridges the gap between:

* Machine transcription ❌
* Human-like language understanding ✅

---

## 📬 Contact

**Pothireddy Nithin Kumar Reddy**
📧 [04pnkr@gmail.com](mailto:04pnkr@gmail.com)

---

## ⭐ How to Use This Project (for GitHub)

* Add demo audio samples
* Add model inference code
* Include results visualization
* Show before vs after text outputs

---
